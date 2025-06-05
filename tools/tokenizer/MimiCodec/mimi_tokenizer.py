import os
import sys
# # Add MimiCodec to the system path
sys.path.append('/home/ydc/musicllm/v2_speech')
# sys.path.clear()

from omegaconf import OmegaConf
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tools.tokenizer.MimiCodec.model.models.MimiCodec import MimiCodec
from tools.tokenizer.abs_tokenizer import AbsTokenizer
import torchaudio


class MimiTokenizer(AbsTokenizer):
    def __init__(self, ckpt_path, device=torch.device('cpu')):
        super(MimiTokenizer, self).__init__()
        #ckpt_path = ckpt_path
        #"/turing_music_fs/music_data/ydc/checkpoints/moshi/tokenizer-e351c8d8-checkpoint125.safetensors"
        #ckpt_path = None
        # GPU is only for offline tokenization
        # So, when distributed training is launched, this should still be on CPU
        self.device = device
        working_dir = os.path.dirname(__file__)
        config_path = os.path.join(working_dir, "mimi_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, "r") as f:
            config = OmegaConf.load(f)
        self.model = MimiCodec(**config.generator.config)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            ckpt_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors")
        load_model(self.model, ckpt_path, strict=False)
        self.model.eval()
        self.sr = 24000
        self.model = self.model.to(self.device)

    def encode(self, wav_root):
        if isinstance(wav_root, str):
            wav, sr = torchaudio.load(wav_root)
            if wav.numel() == 0:
                return None
            if sr != self.sr:
                wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
            wav = wav.unsqueeze(1).to(self.device)
        else:
            wav = wav_root
        with torch.no_grad():
            codes = self.model.encode(wav)
        codes = codes.squeeze(0) #.detach().cpu() # reduce the save space.
        return codes

    def find_length(self, x):
        return x.shape[1]

    def tokenize2(self, token):
        if isinstance(token, torch.Tensor):
            return token.to(torch.int64).transpose(0, 1)
        else:
            raise NotImplementedError
    
    def tokenize(self, wav, sample_rate=24000):
        if isinstance(wav, str):
            # if x is the wave path
            return self.encode(wav)
        elif isinstance(wav, torch.Tensor):
            if wav.dim() == 1: # already done offline
                return wav 
            if wav.dim() == 2: # transfer to 3 dim
                if wav.numel() == 0:
                    return None
                if sample_rate != self.sr:
                    wav = torchaudio.transforms.Resample(sample_rate, self.sr)(wav)
                wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
            wav = wav.to(self.device)
            with torch.no_grad():
                codes = self.model.encode(wav)
            codes = codes.squeeze(0).detach().cpu().to(torch.int16) # reduce the save space.
            return codes
        else:
            raise NotImplementedError

    def detokenize(self, codes):
        #assert codes.shape[0] == 8
        codes = codes.unsqueeze(0)
        wav = self.model.decode(codes)
        wav = wav.squeeze(1).detach().cpu()
        return wav
    

if __name__ == '__main__':
    tokenizer = MimiTokenizer(device=torch.device('cuda:0')).cuda()
    test_wav2 = '/data4/ydc/tts_data/libritts/LibriTTS/test-clean/121/121726/121_121726_000000_000000.wav'
    wav, sr = torchaudio.load(test_wav2)
    if sr != 24000:
        wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    wav = wav.unsqueeze(0).cuda()
    codes = tokenizer.tokenize(wav, 24000)
    print('codes 2', codes.shape)
    wav = tokenizer.detokenize(codes.long().cuda())
    torchaudio.save('sound1.wav', wav, 24000)
