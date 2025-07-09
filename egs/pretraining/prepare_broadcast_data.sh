stage=7
stop_stage=7
ngpu=32


db_root=/mnt/file-201-user-disk-m/cpii.local/dcyang/zjk/data/podcast/podcast

processed_metadata_root=/mnt/file-201-data-disk-m/cpii.local/dcyang/processed/podcast_metadata
processed_audio_root=/mnt/file-201-data-disk-m/cpii.local/dcyang/processed/podcast_data_processed


export CUDA_VISIBLE_DEVICES=2,7
available_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Available GPUs: $available_gpus"

# conda init bash
source /mnt/file-201-user-disk-m/cpii.local/dcyang/zjk/softwares/miniconda3/etc/profile.d/conda.sh

conda activate RSTnet
export PYTHONPATH=$PYTHONPATH:/mnt/file-201-user-disk-m/cpii.local/dcyang/zjk/projects/CSM

mkdir -p $processed_metadata_root
wav_scp=$processed_metadata_root/wav.scp;

# Prepare wav.scp
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    [[ -f "$wav_scp" ]] && rm $wav_scp
    echo "Prepare RSTnet dataset"
    id=0
    find . "${db_root}" -type f -name "*.mp3" | while read -r wav_file; do
        if [[ $wav_file =~ \([0-9]+\)\.mp3$ ]]; then 
            continue
        else
            echo -e "$id\t$wav_file" >> $wav_scp
            id=$((id + 1))
        fi
    done
    find . "${db_root}" -type f -name "*.m4a" | while read -r wav_file; do
        if [[ "$wav_file" =~ \([0-9]+\)\.m4a$ ]]; then
            continue
        else
            echo -e "$id\t$wav_file" >> $wav_scp
            id=$((id + 1))
        fi
    done

fi

# Split the $processed_metadata_root for $ngpu GPUs
# This is done before $processed_metadata_root preprocessing such that multiple GPUs can be used for $processed_metadata_root preprocessing
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the $processed_metadata_root for $ngpu GPUs"
    mkdir -p $processed_metadata_root/${ngpu}splits
    # extra shuf to ensure balance across GPUs
    # So the generated $processed_metadata_root cannot be reproduced due to the shuffle randomness
    if [ -f $processed_metadata_root/wav.scp.shuf ]; then
        rm -f $processed_metadata_root/wav.scp.shuf
    fi
    
    cat $processed_metadata_root/wav.scp | shuf >  $processed_metadata_root/wav.scp.shuf
    split_scp=
    for n in `seq 1 $ngpu`; do
        split_scp="$split_scp $processed_metadata_root/${ngpu}splits/wav.${n}.scp"
    done
    utils/split_scp.pl $processed_metadata_root/wav.scp.shuf $split_scp
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Emilia pipeline: Spk -> VAD -> ASR"
    conda activate AudioPipeline
    utils/run.pl JOB=1:${npgu}  $processed_metadata_root/${ngpu}splits/log/emilia.JOB.log \
    python data_scripts/emilia/main.py \
        --rank JOB \
        --input_scp $processed_metadata_root/${ngpu}splits/wav.JOB.scp \
        --input_folder_path $db_root \
        --output_scp $processed_metadata_root/${ngpu}splits/wav_seg.JOB.scp \
        --output_utt2json $processed_metadata_root/${ngpu}splits/utt2json.JOB \
        --output_folder_path $processed_audio_root \
        --max_duration 120 \
        --config_path data_scripts/emilia/config.json
fi

wait

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Audio Tokenization"
    conda activate RSTnet
    utils/run.pl JOB=1:$ngpu $processed_metadata_root/${ngpu}splits/log/mimi.JOB.log \
        python3 data_scripts/offline_codec_tokenization.py \
            --input-file  $processed_metadata_root/${ngpu}splits/wav_seg.JOB.scp \
            --output-file  $processed_metadata_root/${ngpu}splits/audio_codec.JOB.pt \
            --tokenizer mimi --rank JOB || exit 1;
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Prepare text sequence"
    conda activate RSTnet
    # ../../tools/kaldi/utils/run.pl JOB=1:$ngpu  $processed_metadata_root/${ngpu}splits/log/text_bpe.JOB.log \
    utils/run.pl JOB=1:$ngpu  $processed_metadata_root/${ngpu}splits/log/text_bpe.JOB.log \
    python  data_scripts/text_tokenization_utt2json.py \
        --rank JOB \
        --input-file  $processed_metadata_root/${ngpu}splits/utt2json.JOB \
        --input-audio $processed_metadata_root/${ngpu}splits/audio_codec.JOB.pt \
        --checkpoint_dir /mnt/file-201-user-disk-m/cpii.local/dcyang/zjk/projects/CSM/ckpts/llama-3.2-3B\
        --output-file $processed_metadata_root/${ngpu}splits/tokens.JOB.pt
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "create $processed_metadata_root json"
    #mkdir -p $processed_metadata_root/${ngpu}splits
    for n in `seq 0 $[$ngpu-1]`; do
    python3 data_scripts/create_data_json.py \
        --task moshi \
        --out-json $processed_metadata_root/${ngpu}splits/broadcast_data.${n}.json \
        --hybrid_seq $processed_metadata_root/${ngpu}splits/tokens.$[$n+1].pt \
        & 
    done; wait
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Dataloader test"
    conda activate RSTnet
    python3 ../../utils/dataloader.py \
        --train_data_jsons $processed_metadata_root/${ngpu}splits/broadcast_data.ALL.json \
        --valid_data_jsons $processed_metadata_root/${ngpu}splits/broadcast_data.ALL.json \
        --audio_tokenizer 'mimi' \
        --empty_token 0 \
        --pad_token 2050 \
        --semantic_eos 0 \
        --text_empty_token 0 \
        --text_pad_token 128002 \
        --parallel_number 33 \
        --checkpoint_path /mnt/file-201-user-disk-m/cpii.local/dcyang/zjk/projects/CSM/ckpts/llama-3.2-3B/lit_model.pth
fi
