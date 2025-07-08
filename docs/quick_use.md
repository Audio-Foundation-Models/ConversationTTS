## Installation

```bash
conda create -n RSTnet python=3.12
conda activate RSTnet
pip install torch==2.5.0 torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/cu121
pip install torchtune
pip install tqdm
pip install librosa==0.9.1
pip install matplotlib
pip install omegaconf 
pip install einops
pip install vector_quantize_pytorch
pip install tensorboard
pip install deepspeed
pip install peft

```

## Zero-shot TTS
the v1 version only supports input one text segment, and generate one audio segment。
See [inference/generator.py](inference/generator.py) for zero-shot TTS.
1. Change `resume` at around Line: 175 into path to the checkpoint file.
2. Download the LLama-3.2 tokenizer checkpoint, and set `text_tokenizer_path` at around Line: 188. Download the MimiCodec checkpoint from [Moshi](https://huggingface.co/kyutai/moshika-pytorch-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors?download=true), and set `audio_tokenizer_path` at around Line: 189.
3. In SPEAKER_PROMPTS at around Line: 202~207, fill in path to the speech prompt of the speaker as well as the corresponding text, so that the model can imitate the style of the target speaker.
3. Fill in the text at Line: 192.
4. run `python inference/generator.py`

## Podcast TTS
See [inference/generator_pod.py](inference/generator_pod.py) for English podcast TTS and [inference/generator_pod_cn.py](inference/generator_pod_cn.py) for Chinese podcast TTS.

1. Change `resume` at around Line: 200 into path to the checkpoint file.
2. Download the LLama-3.2 tokenizer checkpoint, and set `text_tokenizer_path` at around Line: 215. Download the MimiCodec checkpoint from [Moshi](https://huggingface.co/kyutai/moshika-pytorch-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors?download=true), and set `audio_tokenizer_path` at around Line: 216.
3. In SPEAKER_PROMPTS at around Line: 186~199, fill in path to the speech prompt of each speaker as well as the corresponding text, so that the model can imitate the style of each speaker.
4. Fill in the `conversation_lines` with the dialogue you want to generate. The speaker alternates between speaker a and speaker b. The first line is spoken by speaker a, the second by speaker b, then speaker a again, and so on. Each line should be a string containing the text of the dialogue.
5. run `python inference/generator_pod.py`


