# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError, FileType
import copy
import json
import librosa
import numpy as np
import sys
import os
import tqdm
import warnings
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
import pyannote.audio
import pandas as pd
import whisperx

from utils.tool import (
    export_to_mp3,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from utils.logger import Logger, time_logger
from models import separate_fast, dnsmos, whisper_asr, silero_vad
from models.speaker_diarization import Diarization3Dspeaker
import glob

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
audio_count = 0


@time_logger
def readaudio(audio_path):
    """
    Read the audio file and convert it to a wav.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        np.ndarray: Audio waveform as a numpy array.
    """
    global audio_count
    name = "audio"
    if audio_path.endswith(".m4a"):
        name = os.path.basename(audio_path)
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
        audio = audio.set_sample_width(2)  # Set bit depth to 16bit
        audio = audio.set_channels(1)  # Set to mono
        waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)/32768.0
        audio = {
            "waveform": waveform,
            "name":name,
            "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
        }
        return audio
    if isinstance(audio_path, str):
        name = os.path.basename(audio_path)
        Audio = pyannote.audio.Audio(cfg["entrypoint"]["SAMPLE_RATE"],"downmix")
    elif isinstance(audio_path, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")
    
    try:
        audio,sample_rate= Audio(audio_path)
        waveform=audio[0].numpy()
        waveform=waveform.astype(np.float32)
    except Exception as e:
        logger.error(f"Error reading audio file using Pyannote: {audio_path}, {e}")
        try:
            # 使用 librosa 加载音频
            waveform, sample_rate = librosa.load(audio_path, sr=cfg["entrypoint"]["SAMPLE_RATE"], mono=True)
        except Exception as e:
            logger.error(f"Failed to load audio using librosa: {e}")
            raise RuntimeError(f"Unable to load audio file: {audio_path}")

    return{
        "waveform": waveform,
        "name": name,
        "sample_rate": sample_rate
    }



@time_logger
def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """

    t_audio=AudioSegment(
        audio["waveform"].tobytes(),
        frame_rate=audio["sample_rate"],
        sample_width=audio["sample_depth"],
        channels=audio["channels"]
    )
    logger.debug("Entering the preprocessing of audio")

    t_audio = t_audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    t_audio = t_audio.set_sample_width(2)  # Set bit depth to 16bit
    t_audio = t_audio.set_channels(1)

    waveform = np.array(t_audio.get_array_of_samples(), dtype=np.float32)/32768.0
    logger.debug(
        f"for test asr > detect_language: audio max: {max(waveform)}"
    )
    language, prob = asr_model.detect_language(waveform)
    logger.debug(
        f"asr > language: {language}, prob: {prob}"
    )


    return{
        "waveform": waveform,
        "name": audio["name"],
        "sample_rate": audio["sample_rate"],
    }


@time_logger
def source_separation(predictor, audio):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A dictionary containing the separated vocals and updated audio waveform.
    """

    mix, rate = None, None

    if isinstance(audio, str):
        mix, rate = librosa.load(audio, mono=False, sr=44100)
    else:
        # resample to 44100
        rate = audio["sample_rate"]
        mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)

    vocals, no_vocals = predictor.predict(mix)

    # convert vocals back to previous sample rate
    logger.debug(f"vocals shape before resample: {vocals.shape}")
    vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    logger.debug(f"vocals shape after resample: {vocals.shape}")
    audio["waveform"] = vocals[:, 0]  # vocals is stereo, only use one channel

    return audio


# Step 2: Speaker Diarization
@time_logger
def speaker_diarization(audio):
    """
    Perform speaker diarization on the given audio using pyannote model.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """
    logger.debug(f"Start speaker diarization")

    waveform=audio['waveform']
    # waveform=waveform.astype(np.float32)
    waveform=torch.tensor(waveform)
    waveform = torch.unsqueeze(waveform, 0)
    segments = dia_pipeline({"waveform": waveform, "sample_rate": audio['sample_rate']})

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")
    # each speaker's number of segments
    speaker_stats = diarize_df.groupby("speaker").size()
    logger.debug(f"speaker_stats: {speaker_stats}")
    return diarize_df


@time_logger
def deduplicate(speakerdia):
    """
    Deduplicate the speaker diarization segments by merging overlapping segments.

    Args:
        speakerdia (list): List of speaker diarization segments with start, end, and speaker labels.

    Returns:
        list: A list of deduplicated speaker diarization segments.
    """
    deduped_segments = []
    last_seg=speakerdia[0]
    for segment in speakerdia:
        if segment["start"] >= last_seg["end"]:
            deduped_segments.append(last_seg)
            last_seg=segment
            continue
        elif segment["end"] <= last_seg["end"]:
            continue
        elif segment["start"] ==last_seg["start"] and segment["end"] > last_seg["end"]:
            last_seg = segment
        elif segment["speaker"] == last_seg["speaker"]:
            last_seg={
                    "start": last_seg["start"],
                    "end": segment["end"],
                    "speaker": last_seg["speaker"],
                }
        else:
            deduped_segments.append(
                {
                    "start": last_seg["start"],
                    "end": segment["start"],
                    "speaker": last_seg["speaker"],
                }
            )
            last_seg=segment
    deduped_segments.append(last_seg)
    return deduped_segments


@time_logger
def cut_by_speaker_label(vad_list):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 0  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 2  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH  # update end time
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start  # update start time
                vad["end"] = segment_end
            updated_list.append(vad)
            continue
        # when the vad list is empty, or the speaker is different, or the vad segment is long enough ,not to merge
        if (
            last_speaker is None #for the first vad
            or last_speaker != vad["speaker"] # different speaker:
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH # vad segment is long enough
        ):
            updated_list.append(vad)
            continue

        # the speaker is the same, if the gap is long enough ,or to be merged segments are too long ,not to merge
        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    return filter_list


@time_logger
def asr(speaker_segment, audio, word_level_timestamp=False):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments of the given audio.

    Args:
        vad_segments (list): List of VAD segments with start and end times.
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        list: A list of ASR results with transcriptions and language details.
    """
    if len(speaker_segment) == 0:
        return []
    temp_audio = audio["waveform"]
    start_time = speaker_segment[0]["start"]
    end_time = speaker_segment[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]  # remove silent start and end

    # update speaker_segment start and end time (this is a little trick for batched asr:)
    for idx, segment in enumerate(speaker_segment):
        speaker_segment[idx]["start"] -= start_time
        speaker_segment[idx]["end"] -= start_time


    if multilingual_flag:
        logger.debug("Multilingual flag is on")
        language, prob = asr_model.detect_language(temp_audio)
        all_transcribe_result = []
        transcribe_result_temp = asr_model.transcribe(
            temp_audio,
            speaker_segment,
            batch_size=batch_size,
            language=language,
            print_progress=False,
        )
        if word_level_timestamp:
            if language not in alignment_models:
                alignment_models[language] = whisperx.load_align_model(language_code=language, device=device)
            model_a, metadata = alignment_models[language]
            result = whisperx.align(transcribe_result_temp["segments"], model_a, metadata, temp_audio, device, return_char_alignments=False)
            result = result["segments"]
            import pdb; pdb.set_trace()
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
                # TODO: align后会重新进行一次分割，需要贪心对齐一下来恢复speaker信息
                # result[idx]["speaker"] = transcribe_result_temp["segments"][idx]["speaker"]
                for word_idx, word in enumerate(segment["words"]):
                    result[idx]["words"][word_idx]["start"] += start_time
                    result[idx]["words"][word_idx]["end"] += start_time
            all_transcribe_result.extend(result)
        else:
            result = transcribe_result_temp["segments"]
            # restore the segment annotation
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
            all_transcribe_result.extend(result)
        # sort by start time
        all_transcribe_result = sorted(all_transcribe_result, key=lambda x: x["start"])
        # reprocess the missing setences

        return all_transcribe_result
    else:
        logger.debug("Multilingual flag is off")
        language, prob = asr_model.detect_language(temp_audio)
        if language in supported_languages and prob > 0.8:
            transcribe_result = asr_model.transcribe(
                temp_audio,
                speaker_segment,
                batch_size=batch_size,
                language=language,
                print_progress=False,
            )
            
            result = transcribe_result["segments"]
            for idx, segment in enumerate(result):
                result[idx]["language"] = transcribe_result["language"]
            return result
        else:
            return []

@time_logger
def mos_prediction(audio, vad_list):
    """
    Predict the Mean Opinion Score (MOS) for the given audio and VAD segments.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        vad_list (list): List of VAD segments with start and end times.

    Returns:
        tuple: A tuple containing the average MOS and the updated VAD segments with MOS scores.
    """
    audio = audio["waveform"]
    sample_rate = 16000

    audio = librosa.resample(
        audio, orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], target_sr=sample_rate
    )

    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio[start:end]

        dnsmos = dnsmos_compute_score(segment, sample_rate, False)["OVRL"]

        vad_list[index]["dnsmos"] = dnsmos

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])

    logger.debug(f"avg predict_dnsmos for whole audio: {predict_dnsmos}")

    return predict_dnsmos, vad_list


def filter(mos_list):
    """
    Filter out the segments with MOS scores, wrong char duration, and total duration.

    Args:
        mos_list (list): List of VAD segments with MOS scores.

    Returns:
        list: A list of VAD segments with MOS scores above the average MOS.
    """
    filtered_audio_stats, all_audio_stats = calculate_audio_stats(mos_list)
    filtered_segment = len(filtered_audio_stats)
    all_segment = len(all_audio_stats)
    logger.debug(
        f"> {all_segment - filtered_segment}/{all_segment} {(all_segment - filtered_segment) / all_segment:.2%} segments filtered."
    )
    filtered_list = [mos_list[idx] for idx, _ in filtered_audio_stats]
    return filtered_list


def merge_segments(
        segments_list, 
        chunk_size, 
        blank_threshold = 3,
        length_threshold = 3,
        ):
    """
    Merge operation described in paper
    """
    curr_end = 0
    merged_segments = []
    seg_idxs = []

    assert chunk_size > 0
    if segments_list is None or len(segments_list) == 0:
        return merged_segments
    # Make sure the starting point is the start of the segment.
    curr_start = segments_list[0]["start"]

    for seg in segments_list:
        # Open a new section
        if (seg["end"] - curr_start > chunk_size) or \
            (seg["start"] - curr_end > blank_threshold):
            # If previous section is not empty, add it to the list
            if curr_end-curr_start > length_threshold:
                merged_segments.append({
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                })
            curr_start = seg["start"]
            seg_idxs = []
        # Add segment to current section
        curr_end = seg["end"]
        seg_idxs.append(seg)
    # add final
    merged_segments.append({ 
                "start": curr_start,
                "end": curr_end,
                "segments": seg_idxs,
            })

    return merged_segments


@time_logger
def main_process(audio_path, save_path=None, audio_name=None):
    """
    Process the audio file, including standardization, source separation, speaker segmentation, VAD, ASR, export to MP3, and MOS prediction.

    Args:
        audio_path (str): Audio file path.
        save_path (str, optional): Save path, defaults to None, which means saving in the "_processed" folder in the audio file's directory.
        audio_name (str, optional): Audio file name, defaults to None, which means using the file name from the audio file path.

    Returns:
        tuple: Contains the save path and the MOS list.
    """
    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")

    # for a single audio from path Ïaaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    save_path = save_path or os.path.join(
        os.path.dirname(audio_path) + "_processed", audio_name
    )
    os.makedirs(save_path, exist_ok=True)
    logger.debug(
        f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}"
    )
    audio_dir=os.path.dirname(audio_path)
    dirs=audio_dir.split('/')
    audio_name=dirs[-1]+audio_name
    # If the audio has already been processed, skip it
    # Check if pre-processed files already exist
    pattern = os.path.join(save_path, "audio", f"{audio_name}_*.mp3")
    existing_audio_files = glob.glob(pattern)
    if existing_audio_files:
        utt2wav = {}
        utt2json = {}
        for audio_file in existing_audio_files:
            uttid = os.path.splitext(os.path.basename(audio_file))[0]
            utt2wav[uttid] = audio_file

            metadata_file = os.path.join(save_path, "metadata", f"{uttid}.json")
            if os.path.exists(metadata_file):
                utt2json[uttid] = metadata_file

        logger.info(f"Found pre-processed files for {audio_path}; skipping processing.")
        return utt2wav, utt2json

    logger.info(
        "Step 1: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
    )

    audio = readaudio(audio_path)

    logger.info("Step 2: Speaker Diarization")
    speakerdia = speaker_diarization(audio)
    # transform dataframe speakerdia to List[dict]
    speakerdia = [
        {
            "start": row["start"],
            "end": row["end"],
            "speaker": row["speaker"],
        }
        for index, row in speakerdia.iterrows()
    ]

    logger.info("Step 3: ASR")
    asr_result = asr(speakerdia, audio, word_level_timestamp=args.word_level_timestamp)

    filtered_list = merge_segments(asr_result, chunk_size=args.max_duration)

    logger.info("Step 4: write result into MP3 and JSON file")
    utt2wav, utt2json = export_to_mp3(audio, filtered_list, save_path, audio_name)

    return utt2wav, utt2json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="rank of the current worker, used for distributed training",
    )
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--input_scp",
        type=str,
        default="",
        help="input wav.scp, this will override config if set",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="",
        help="output folder path, default: <input_folder_path>_processed",
    )
    parser.add_argument(
        "--output_scp",
        type=str,
        default="",
        help="output wav.scp"
    )
    parser.add_argument(
        "--output_utt2json",
        type=str,
        default="",
        help="output utt2json"
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=60,
        help="max duration for a single session",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="The compute type to use for the model",
    )
    parser.add_argument(
        "--whisper_arch",
        type=str,
        default="medium",
        help="The name of the Whisper model to load.",
    )
    parser.add_argument(
        "--word_level_timestamp",
        action="store_true",
        help="Whether to return word level timestamps from the ASR model.",
    )
    parser.add_argument("--seg-len", type=int, default=105)
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="The number of CPU threads to use per worker, e.g. will be multiplied by num workers.",
    )
    parser.add_argument(
        "--exit_pipeline",
        type=bool,
        default=False,
        help="Exit pipeline when task done.",
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    cfg = load_cfg(args.config_path)

    logger = Logger.get_logger()

    if args.input_folder_path:
        logger.info(f"Using input folder path: {args.input_folder_path}")
        cfg["entrypoint"]["input_folder_path"] = args.input_folder_path

    logger.debug("Loading models...")

    # Load models
    if detect_gpu():
        logger.info("Using GPU")
        device_name = f"cuda"
        
        rank = args.rank - 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
        max_gpu = torch.cuda.device_count()
        rank = (rank % max_gpu)
        device = torch.device(f"cuda:{rank}")

    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)
        # whisperX expects compute type: int8
        logger.info("Overriding the compute type to int8")
        args.compute_type = "int8"

    check_env(logger)

    # Speaker Diarization
    logger.debug(" * Loading Speaker Diarization Model")
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError(
            "huggingface_token must start with 'hf', check the config file. "
            "You can get the token at https://huggingface.co/settings/tokens. "
            "Remeber grant access following https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#tldr"
        )
    dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=cfg["huggingface_token"],
    )
    dia_pipeline.to(device)

    # Use camplus for speaker diarization
    # TODO: Enable overlap
    # diarization = Diarization3Dspeaker(device, include_overlap=True, hf_access_token=cfg["huggingface_token"])

    # ASR
    logger.debug(" * Loading ASR Model")
    asr_model = whisper_asr.load_model(
        args.whisper_arch,
        device_name,
        device_index=rank,
        compute_type=args.compute_type,
        threads=args.threads,
        asr_options={
            "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음.",
            "word_timestamps": True,
        },
    )
    if args.word_level_timestamp:
        alignment_models = {}

    # VAD
    logger.debug(" * Loading VAD Model")
    # vad = silero_vad.SileroVAD(device=device)
    vad=None

    # Background Noise Separation
    logger.debug(" * Loading Background Noise Model")
    separate_predictor1 = separate_fast.Predictor(
        args=cfg["separate"]["step1"], device=device_name
    )

    # DNSMOS Scoring
    logger.debug(" * Loading DNSMOS Model")
    primary_model_path = cfg["mos_model"]["primary_model_path"]
    dnsmos_compute_score = dnsmos.ComputeScore(primary_model_path, device_name)
    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    logger.debug(f"supported languages multilingual {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")

    input_folder_path = cfg["entrypoint"]["input_folder_path"]

    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    if args.input_scp:
        logger.info(f"Using input wav.scp: {args.input_scp}")
        audio_paths = []
        with open(args.input_scp) as f:
            for line in f:
                line = line.strip()
                audio_path = line.split('\t')[-1]
                audio_paths.append(audio_path)
                # audio_paths.append(os.path.join(input_folder_path, audio_path))
    else:
        audio_paths = get_audio_files(input_folder_path)  # Get all audio files
    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path}")

    utt2wav, utt2json = {}, {}
    for path in audio_paths:
        _utt2wav, _utt2json = main_process(path, args.output_folder_path)
        utt2wav.update(_utt2wav)
        utt2json.update(_utt2json)
    
    if args.output_scp:
        with open(args.output_scp, "w") as f:
            for k, v in utt2wav.items():
                f.write(f"{k}\t{v}\n")

    if args.output_utt2json:
        with open(args.output_utt2json, "w") as f:
            for k, v in utt2json.items():
                f.write(f"{k}\t{v}\n")
