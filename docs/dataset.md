## Main idea

## 0. Environment Setup
You must use python>=3.10

>1.install python and cuda

>2.install packages
```
conda create -y -n RSTnet python=3.9 
conda activate RSTnet

cd egs/pretraining/data_scripts/emilia
bash env.sh
```
>3.Manually download the checkpoints of UVR-MDX-NET-Inst_HQ_3 [UVR-MDX-NET-Inst_3.onnx](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx) and DNSMOS P.835 [sig_bak_ovr.onnx](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx), then save their path for the next step configuration (i.e. #2 and #3 TODO).

>4.Creat the access token to pyannote/speaker-diarization-3.1 following the [guide](https://huggingface.co/pyannote/speaker-diarization-3.1#requirements), then save it for the next step configuration.

>Make sure you have stable connection to GitHub and HuggingFace. The checkpoints of Silero and Whisperx-medium will be downloaded automatically on the pipeline's first run.

## 1. Data Preparation


### 1.1 Broadcast Data
This Data processing methodology derived from Emilia with some modifications.
The Pipeline includes the following major steps:

Standardization：Audio normalization

Speaker Diarization: Get medium-length single-speaker speech data

ASR: Get transcriptions of the speech segments

Filtering: Obtain the final processed dataset

#### 1.Config File

The config.json can be found at `data_scripts/emilia`. Make sure RSTnet environments are ready.
```json
{
    "language": {
        "multilingual": true,
        "supported": [
            "zh",
            "en",
            "fr",
            "ja",
            "ko",
            "de"
        ]
    },
    "entrypoint": {
        // TODO: Fill in the input_folder_path. 
        "input_folder_path": "examples", // #1: Data input folder for processing
        "SAMPLE_RATE": 16000
    },
    "separate": {
        "step1": {
            // TODO: Fill in the source separation model's path. 
            "model_path": "/path/to/model/separate_model/UVR-MDX-NET-Inst_HQ_3.onnx", // #2: Model path
            "denoise": true,
            "margin": 44100,
            "chunks": 15,
            "n_fft": 6144,
            "dim_t": 8,
            "dim_f": 3072
        }
    },
    "mos_model": {
        // TODO: Fill in the DNSMOS prediction model's path. 
        "primary_model_path": "/path/to/model/mos_model/DNSMOS/sig_bak_ovr.onnx" // #3: Model path
    },
     // TODO: Fill in your huggingface access token for pynannote. 
    "huggingface_token": "<HUGGINGFACE_ACCESS_TOKEN>" // #4: Huggingface access token for pyannote
}
```
#### 2.Run Scripts


see [egs/pretraining/prepare_broadcast_data.sh](egs/pretraining/prepare_broadcast_data.sh) for data preparation. This bash controls the data pipeline . You can run the program step by step via changing  `stage` and `stop stage` to see how it works. This bash takes inputs and outputs as follow:

>Input: A folder containing conversational waveforms.  
Output (ALL=0 ~ ngpu-1):  
broadcast_data.ALL.json: Metadata containing paths to `tokens.ALL+1.pt`. Provided as  `--train_data_jsons` / `--valid_data_jsons` parameters when training.  
tokens.ALL+1.pt: Dict of interleaved text-audio codes. utt_name: Torch.tensor with shape [33, T_audio] and type torch.int16.

0. Fill `CUDA_VISIBLE_DEVICES` with your available GPUs, your data will be seperated into `ngpu` parts and distributed to GPUs evenly. 

1. Put all your audio data under the directory: `db_root`, audios in `mp3` ,`wav` and `mp4` format are supported. Set `processed_metadata_root` and `processed_audio_root` as the output directory.
```bash
db_root=<path-to-source-wavs>
processed_metadata_root=<path-to-dump-metadata>
processed_audio_root=<path-to-dump-processed-audio>
```
2. set `conda.sh` and `PYTHONPATH` to initial your conda environment correctly.
```bash
source <conda-dir>/etc/profile.d/conda.sh
export PYTHONPATH=$PYTHONPATH:<path-to-RSTnet/MLLM_v2>
```
3. set other parameters like `checkpoint_dir` and `max_duration` with the value you like.
```bash
--checkpoint_dir <path-to-pretrained-LLM>

# hyperparameters
--max_duration: maximum length of a sample session in seconds. Should properly set so as to not exceed GPU memory limit. Default: 120
```


4. Finally, we can run `prepare_broadcast_data.sh`. The dataloader will yield a token sequence with the following [B, 33, t_text+t_audio+2] shape:
```
<|begin_of_text|>[<spk_id1>] <text> ··· [<spk_id1>] <text><|end_of_text|><|text_emply_token|>···
                                                                         <semantic_tokens>···
                                                                         <acoustic_tokens>···
```
In the last stage, we provide codes for dataloader sainity check. If all things works fine, it will output the text stream in the terminal and the audio stream as a .wav file under `egs/pretraining/data_scripts`. Just check if they are aligned.

#### 3. Check Intermediate Result
Intermediate results and error logs are under the directory `processed_metadata_root`
 `processed_audio_root` contains sliced audio files named `original_file_name_x.mp3`, `x` reprensents the xth slice of original audio. `processed_metadata_root` contains the result json files including speaker labels and corresponding transition. The json files are organized as below:
 ```json
 {
    "start": 2213.6753125,
    "end": 2234.973312500002,
    "segments": [
        {
            "text": "我觉得这个问题其实是一个很有意思的问题,因为我觉得它的答案其实是两种,就是逼自己一把和放自己一马,这两种方式都可以。",
            "start": 2213.6753125,
            "end": 2223.8833125,
            "speaker": "SPEAKER_00",
            "language": "zh"
        },
        {
            "text": "但到底好的那个成绩,对,它对你是有着决定性的作用的。这个exactly就要发生在我身上的,因为我记得特别清楚,因为我初三不是就要考上四中拓高升,那一个暑假。",
            "start": 2223.8833125,
            "end": 2234.9733125000002,
            "speaker": "SPEAKER_00",
            "language": "zh"
        }

    ]
}
 ```


### 1.2 TTS Data

TTS metadata vastly differs from each other. We provide an example of processing [a huggingface cantonese dataset](https://huggingface.co/datasets/alvanlii/cantonese-youtube) in `prepare_hf_tts_data.sh`.

>Input: A folder containing .parquet files.  
Output (ALL=0 ~ ngpu-1):  
tts_cantonese_data.ALL.json: Metadata containing paths to `audio_codec.ALL+1.pt` and `text.ALL+1.pt`. Provided as  `--train_data_jsons` / `--valid_data_jsons` parameters when training.  
audio_codec.ALL+1.pt: Dict of mimicodec audio codes. utt_name: Torch.tensor with shape [32, T_audio] and type torch.int16.
text.ALL+1.pt: Dict of text token IDs. utt_name: Torch.tensor with shape [T_text, ] and type torch.int32.

As in 1.1, we need to modify paths and hyperparameters in `prepare_hf_tts_data.sh`.

```bash
# line 5~6
db_root=<path-to-source-wavs>
processed_root=<path-to-dump-outputs>

# line 12~16
source <conda-dir>/etc/profile.d/conda.sh
export PYTHONPATH=$PYTHONPATH:<path-to-RSTnet/MLLM_v2>

# line 66 & 88
--checkpoint_dir <path-to-pretrained-LLM>
```
