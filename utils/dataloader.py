
import json
import os
import sys
import torch
import copy
import random
import logging
import torch.distributed as dist
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer
from utils.task_definition import (
    load_data_for_all_tasks,
    task_formats
)

def print_log(content: str):
    logging.info(content)
    print(content)

def build_data_iterator(
        data_dict,
        tokenizers,
        delay_step=1,
        max_length=-1,
        min_length=-1,
        batch_scale=1000,
        is_train=True,
        n_worker=1,
        seed=999,
        minibatch_debug=-1,
        parallel_number=5,
        text_empty_token = 0,
        semantic_empty_token=0,
        semantic_pad_token = 2049,
        semantic_eos = 10000,
        text_pad_token=128003,
    ):
    find_all_length(data_dict, tokenizers) # get the length
    valid_utts = filter_data(data_dict, max_length, min_length) 
    batches = batchfy(data_dict, valid_utts, batch_scale) # prepare batch
    logging.info(f"Finish pre-process all data. {len(valid_utts)} examples and {len(batches)} batches")
    all_data_dict = {}
    all_data_dict.update(data_dict)
    if minibatch_debug > 0:
        batches = batches[:min(minibatch_debug, len(batches))]
        logging.info(f"only use {len(batches)} as this is a debug mode")
    dataset = Dataset(batches, all_data_dict)
    sampler = DDPSyncSampler(size=len(batches), seed=seed, is_train=is_train)
    # Build iterator. No multi-process when debug
    collate_fn = Collate_Fn_Factory(
            tokenizers = tokenizers,
            max_length=max_length if max_length > 0 else 15000,
            delay_step=delay_step, 
            parallel_number = parallel_number,
            text_empty_token = text_empty_token,
            semantic_empty_token=semantic_empty_token,
            semantic_pad_token = semantic_pad_token,
            semantic_eos = semantic_eos,
            text_pad_token=text_pad_token
    )
    if minibatch_debug != -1:
        iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
        logging.info("disable multi-processing data loading: debug mode")
    else:
        # debug 
        iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=n_worker,
            collate_fn=collate_fn,
        )
    return iterator


def filter_data(data_dict, max_length, min_length):
    # we find the valid key rather than remove the whole exmaple as the invalid exmaples can 
    # also work as the prompt
    keys = list(data_dict.keys())
    if max_length <= 0 and min_length <= 0:
        return keys

    valid_keys = []
    if max_length > 0:
        for k in keys:
            if  (data_dict[k]['length'] <= max_length or max_length <= 0) \
            and (data_dict[k]['length'] >= min_length or min_length <= 0):
                valid_keys.append(k)
    logging.info(f"you requires length between [{min_length}, {max_length}] so only {len(valid_keys)} examples are reserved.")
    return valid_keys

def find_all_length(data_dict, tokenizers):
    """ length found here is only for batchfy. it is not the real length as there may be more special tokens """
    for example_id, d in data_dict.items():
        data_format = task_formats[d['task']]
        length = 0
        for key, key_type in zip(data_format['loss_key'], data_format['type']):
            if key_type == 'hybrid':
                this_length = d[key].shape[-1]
            else:
                this_length = tokenizers[key_type].find_length(d[key])
            length += this_length
        d['length'] = length

def batchfy(data_dict, batch_utts,  batch_scale):
    # we should make sure each batch includes at least one text-only?
    ''' we sort the batch for text-only and others respectively. 8B llama3 support batch scale 2500
        Then, we make sure the text-only data is always exists in the batch. 
    '''
    batch_utts.sort(key=lambda x: data_dict[x]['length']) # sort audio-related data
    batch_lengths = [data_dict[k]['length'] for k in batch_utts] # 

    # Only take care of the uttid rather than the whole example
    batches, batch, summed_tokens = [], [], 0
    idx = 0
    for utt, l in zip(batch_utts, batch_lengths):
        if l + summed_tokens > batch_scale:
            assert len(batch) > 0, f"batch_tokens should be larger: {batch_scale}"
            batches.append(copy.deepcopy(batch))
            batch, summed_tokens = [], 0
        summed_tokens += l
        batch.append(utt)

    if len(batch) > 0:
        batches.append(copy.deepcopy(batch))

    # TODO: maybe report statistics
    logging.info(f'After batchfy, there are {len(batches)} batches')
    return batches 


class Dataset(torch.utils.data.Dataset):
    """ Dataset. Each example is exactly a batch """
    def __init__(self, data_split, data_dict):
        self.data_split = data_split # batches
        self.data_dict = data_dict

    def __getitem__(self, index):
        uttids = self.data_split[index]
        return [(uttid, self.data_dict[uttid]) for uttid in uttids]

    def __len__(self):
        return len(self.data_split)

class SequentialSampler(object):
    def __init__(self, sequence):
        self.seq = sequence

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def refresh(self):
        pass

class DDPSyncSampler(object):
    def __init__(self, size, seed, is_train=True):
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.is_train = is_train

        # Ensure that data iterator aross all GPUs has the same number of batches
        if dist.is_initialized() and torch.cuda.is_available():
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            size = torch.Tensor([size]).to(device).int()
            dist.all_reduce(size, dist.ReduceOp.MAX)

            self.pad_number = size.item() - self.size
            self.rank = dist.get_rank()
        else:
            logging.warning("torch.distributed is not available!")
            self.pad_number = 0
            self.rank = 0

        self.refresh()

    def refresh(self):
        seq = list(range(self.size))

        if self.is_train:
            # Assume the batches are sorted from shortest to longest
            # This introduces local randomness by local random shuffling
            # otherwise each global batch will be identical across epochs
            chunk_size, start = 10, 0
            random.seed(self.rank + self.seed + self.epoch)
            while start < self.size:
                seg = seq[start: min(self.size, start + chunk_size)]
                local_random_order = random.sample(list(range(len(seg))), len(seg))
                seg = [seg[i] for i in local_random_order]
                seq[start: min(self.size, start + chunk_size)] = seg
                start += len(seg)

            # even after this shuffle, the batch lengths across GPUs 
            # are very similar
            random.seed(self.seed + self.epoch)
            random.shuffle(seq)

        # so the #batches are identical across GPUs
        if self.pad_number > 0:
            seq = list(range(self.pad_number)) + seq

        self.seq = seq
        self.epoch += 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def get_state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'seed': self.seed,
        }
        return state_dict

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class Collate_Fn_Factory(object):
    ''' We need to carefully define our special tokens
        Empty token must different with padding tokens.
        llama3 tokenizer: reserved tokens from 128002-128255
        Shape: (B, T, L) the L-th denotes the text streaming
    '''
    def __init__(self, 
                 tokenizers=None,
                 max_length=15000,
                 delay_step=1,
                 parallel_number = 2,
                 text_empty_token = 128002,
                 text_pad_token = 0,
                 semantic_empty_token = 0,
                 semantic_pad_token = 16385,
                 semantic_eos = 10001,
    ):
        self.max_length = max_length
        self.delay_step = delay_step
        self.text_empty_token = text_empty_token
        self.text_pad_token = text_pad_token # ?
        self.semantic_empty_token = semantic_empty_token
        self.semantic_pad_token = semantic_pad_token
        self.semantic_eos = semantic_eos
        self.parallel_number = parallel_number # how many parallel tokens
        self.tokenizers = tokenizers
        
    def text_pad(self, x):
        '''input 1-dimension sequence. add empty token for semantic streaming.
        '''
        sequences = torch.ones((len(x), self.parallel_number)).to(torch.int64)
        sequences[:, -1] = x # the text tokens
        sequences[:,:-1] = sequences[:,:-1]*self.semantic_empty_token # we will set as 0
        return sequences

    def audio_pad(self, x):
        '''input audio (T, 4) sequence. Add empty token for text.
        '''
        sequences = torch.ones((x.shape[0], self.parallel_number)).to(torch.int64)*self.text_empty_token
        sequences[:,:-1] = x 
        return sequences

    def splice_sequence(self, d, keys, types, loss_key):
        start =  0
        task = d['task']
        if task == 'text_only': # not used now 
            print('not implement now')
            assert 1==2

        elif task == 'audio_only': # not used now 
            print('not implement now')
            assert 1==2
        
        elif task == 'moshi':   # now for sentence level text-audio interleaved
            this_data = d['hybrid_seq'].to(torch.int64).transpose(0, 1)  # [N, T] -> [T, N]
            # Padding has been done in text_tokenization_utt2json.py, we only need to generate the mask
            this_mask = torch.ones((this_data.shape[0], self.parallel_number), dtype=torch.bool)
            zero_rows_torch = torch.all(this_data == 0, dim=1)  # 
            zero_row_indices_torch = torch.where(zero_rows_torch)[0]
            this_mask[this_data==self.semantic_empty_token] = False
            this_mask[this_data==self.text_empty_token] = False
            this_mask[this_data==self.semantic_pad_token] = False
            this_mask[this_data==self.text_pad_token] = False
            this_mask[zero_row_indices_torch,:-1] = True # we should set the stop token as true for training
        
        elif task == 'musicllm_v1' or task == 'speechllm_v1':
            this_text_data = self.tokenizers['text'].tokenize2(d['text_seq'])
            
            this_audio_data = self.tokenizers['audio'].tokenize2(d['audio_seq'])
            eos_frame = torch.ones(1, this_audio_data.shape[1])*self.semantic_eos # add eos tokens
            this_audio_data = torch.cat([this_audio_data, eos_frame], dim=0) # (T+1,4)

            this_text_data = self.text_pad(this_text_data)
            this_text_mask = torch.zeros((this_text_data.shape[0], self.parallel_number))
            this_text_mask[:,-1] = True
            this_audio_data = self.audio_pad(this_audio_data)
            this_audio_mask = torch.zeros((this_audio_data.shape[0], self.parallel_number))
            this_audio_mask[:,:-1] = True 
            this_data = torch.cat([this_text_data, this_audio_data], dim=0) # combine along time
            this_mask = torch.cat([this_text_mask, this_audio_mask], dim=0)
            
        else:
            raise NotImplementedError(args.audio_tokenizer)
        start = this_data.shape[0] # the length of sequence
        return this_data, this_mask, start

    def init_sequence(self, batch_size):
        sequences = torch.ones((batch_size, self.max_length+2, self.parallel_number, )).long() 
        sequences[:,:,-1] = sequences[:,:,-1]*self.text_pad_token
        sequences[:,:,:-1] = sequences[:,:,:-1]*self.semantic_pad_token
        return sequences

    def decoder_only_collate_fn(self, batch):
        """Output: data and mask [B, T, L] """
        batch_size = len(batch)
        sequences = self.init_sequence(batch_size)
        masks = torch.zeros((batch_size, self.max_length+2, self.parallel_number)) #.bool() # record the loss weight
        lengths, example_ids= [], []
        for idx, (example_id, d) in enumerate(batch):
            task_format = task_formats[d['task']]
            sequence, mask, length = self.splice_sequence(d, task_format['keys'], task_format['type'], task_format['loss_key'])

            sequences[idx, :sequence.shape[0], :] = sequence
            masks[idx, :mask.shape[0], :] = mask # we donot calculate loss for PADING part
            lengths.append(length)
            example_ids.append(example_id)

        sequences = sequences[:, :max(lengths), :].long() # 
        masks = masks[:, :max(lengths), :]
        lengths = torch.Tensor(lengths).long()

        return sequences, masks, lengths, example_ids

    def __call__(self, batch):
        assert len(batch) == 1, "batch size should only be 1"
        batch = batch[0] # a list of data
        return self.decoder_only_collate_fn(batch)

def get_data_iterator_tokenizer_vocabulary(
        args,
        train_jsons,
        valid_jsons,
        batch_scale=3000,
        delay_step=1,
        minibatch_debug=-1,
        max_length=-1,
        min_length=-1,
        non_acoustic_repeat=1,
        n_worker=4,
        decoder_only=True,
        parallel_number=9,
        text_empty_token = 128002,
        semantic_empty_token=2048,
        semantic_pad_token = 2049,
        semantic_eos = 10000,
        text_pad_token=128003,
        seed=999
    ):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    # (1) load all data in the raw format
    logging.info(f"loading train: {train_jsons}")
    train_data_dict, train_text_dict = load_data_for_all_tasks(train_jsons)
    # print('train_data_dict ', len(train_data_dict.keys()), len(train_text_dict.keys()))
    logging.info(f"loading valid:  {valid_jsons}")
    valid_data_dict, valid_text_dict = load_data_for_all_tasks(valid_jsons)
    # print('train_data_dict ', len(valid_data_dict.keys()), len(valid_text_dict.keys()))
    tokenizers = {}
    if args.audio_tokenizer is not None and args.audio_tokenizer != "none":
        if args.audio_tokenizer == "semantic":
            audio_tokenizer = None
        elif args.audio_tokenizer == 'mimi':
            audio_tokenizer = MimiTokenizer()
        else:
            raise NotImplementedError(args.audio_tokenizer)
        tokenizers['audio'] = audio_tokenizer
    else:
        audio_tokenizer = None
        logging.info(f"Did not build audio tokenizer: {args.audio_tokenizer}")
    if args.text_tokenizer is not None and args.text_tokenizer != "none":
        if args.text_tokenizer == 'llama3-8B' or args.text_tokenizer == 'qwen':
            text_tokenizer = TextTokenizer(os.path.dirname(args.checkpoint_path))
        else:
            raise NotImplementedError(args.text_tokenizer)
        tokenizers['text'] = text_tokenizer
    else:
        text_tokenizer = None
        logging.info(f"Did not build audio tokenizer: {args.text_tokenizer}")
    # (2) build data iterator
    valid_iterator = build_data_iterator(
        valid_data_dict,
        tokenizers,
        delay_step=delay_step, 
        max_length=max_length,
        min_length=min_length,
        batch_scale=batch_scale,
        is_train=False,
        n_worker=n_worker,
        seed=seed,
        minibatch_debug=minibatch_debug,
        parallel_number = parallel_number,
        text_empty_token = text_empty_token,
        semantic_empty_token = semantic_empty_token,
        semantic_pad_token = semantic_pad_token,
        semantic_eos = semantic_eos,
        text_pad_token = text_pad_token,
    )
    train_iterator = build_data_iterator(
        train_data_dict, 
        tokenizers,
        delay_step=delay_step, 
        max_length=max_length,
        min_length=min_length,
        batch_scale=batch_scale, 
        is_train=True,
        n_worker=n_worker,
        seed=seed,
        minibatch_debug=minibatch_debug,
        parallel_number=parallel_number,
        text_empty_token = text_empty_token,
        semantic_empty_token = semantic_empty_token,
        semantic_pad_token = semantic_pad_token,
        semantic_eos = semantic_eos,
        text_pad_token = text_pad_token
    )
    logging.info('all iterator built')
    return train_iterator, valid_iterator

if __name__ == "__main__":
    # get_data_iterator_tokenizer_vocabulary(sys.argv[1:2], sys.argv[2:3], n_worker=1) 
    from utils.arguments import get_args
    from utils.train_utils import find_data_jsons
    import torchaudio
    args = get_args()
    train_iter, valid_iter = get_data_iterator_tokenizer_vocabulary(
        args, 
        find_data_jsons(args.train_data_jsons, rank=0, world_size=1), 
        find_data_jsons(args.valid_data_jsons, rank=0, world_size=1), 
        n_worker=0,
        parallel_number=args.parallel_number,
        text_empty_token=args.text_empty_token,
        semantic_empty_token=args.empty_token,
        semantic_pad_token=args.pad_token,
        semantic_eos=args.semantic_eos,
        text_pad_token=args.text_pad_token,
        )
    
    if args.audio_tokenizer is not None and args.audio_tokenizer != "none":
        if args.audio_tokenizer == "semantic":
            audio_tokenizer = None
        elif args.audio_tokenizer == 'mimi':
            audio_tokenizer = MimiTokenizer()
        else:
            raise NotImplementedError(args.audio_tokenizer)
    else:
        audio_tokenizer = None
        logging.info(f"Did not build audio tokenizer: {args.audio_tokenizer}")
    if args.text_tokenizer is not None and args.text_tokenizer != "none":
        if args.text_tokenizer == 'llama3-8B' or args.text_tokenizer == 'qwen':
            text_tokenizer = TextTokenizer(os.path.dirname(args.checkpoint_path))
        else:
            raise NotImplementedError(args.text_tokenizer)



    for i, batch in enumerate(train_iter):
        if i > 10:
            break

        utt_id = batch[3][0]
        length = batch[2][0]
        text = text_tokenizer.decode(batch[0][0, :length, -1])
        audio = audio_tokenizer.detokenize(batch[0][0, :length, :-1].T)
        print(f"ID: {utt_id}, Text: {text}")
        torchaudio.save(f'{utt_id}_out.wav', audio, 24000)
        print(batch)

    for i, batch in enumerate(valid_iter):
        if i > 10:
            break

        utt_id = batch[3][0]
        length = batch[2][0]
        text = text_tokenizer.decode(batch[0][0, :length, -1])
        audio = audio_tokenizer.detokenize(batch[0][0, :length, :-1].T)
        print(f"ID: {utt_id}, Text: {text}")
        torchaudio.save(f'{utt_id}_out.wav', audio, 24000)
        print(batch)
        
