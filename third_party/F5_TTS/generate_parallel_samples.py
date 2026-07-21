from importlib.resources import files
from f5_tts.api import F5TTS
from tqdm import tqdm
import glob
import argparse
import os
import torch
import librosa
from pathlib import Path


def save_synthesis(f5tts: F5TTS, wav_path: str, device: str):
    ref_text = "for the twentieth time that evening the two men shook hands."
    ref_file = "/data/waris/data/AccentClassificationData/train/american/BDL/wav/arctic_a0003.wav"
    subject_id = Path(wav_path).parent.parent.name
    out_wav_path = Path(wav_path).parent.parent.parent.parent / 'parallel_data' / 'native_f5tts' / subject_id / 'wav' / Path(wav_path).name
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if out_wav_path.exists():
        return
    text_fpath =  Path(wav_path).parent.parent / 'transcript' / (Path(wav_path).stem + '.txt')

    gen_text = open(text_fpath, 'r', encoding="utf-8").readlines()[0].strip().lower()

    wav, sr, spec = f5tts.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave=str(out_wav_path),
        file_spec=None,
        seed=None,
        fix_duration=librosa.get_duration(filename=wav_path) + 3,  # add 3 seconds buffer to ensure no truncation
    )

def get_model(device="cuda"):
    f5tts = F5TTS(device=device)
    return f5tts

def shard_by_rank(items, rank, world_size):
    return items[rank::world_size]


def run_rank(pattern: str, rank: int, world_size: int, device: str):
    wav_files = sorted(glob.glob(pattern, recursive=True))
    if not wav_files:
        print(f"[rank {rank}] No files matched pattern: {pattern}")
        return

    shard = shard_by_rank(wav_files, rank, world_size)
    model = get_model(device=device)

    pbar = tqdm(shard, desc=f"[rank {rank}] encoding", unit="file")
    for fp in pbar:
        try:
            save_synthesis(model, fp, device)
        except Exception as e:
            pbar.write(f"[rank {rank}] ERROR {fp}: {e}")



def main():
    parser = argparse.ArgumentParser(description="multi-GPU speech synthesis")
    parser.add_argument("--dataset", type=str, default="AESRC")
    parser.add_argument("--pattern", type=str, default="/data/mtseng/voice_datasets/AccentedEnglish/Spanish_curated/TODO/data/**/wav/*.wav",
                        help="Glob pattern for WAV files (recursive OK)")
    args = parser.parse_args()

    # Multi-GPU via torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Device resolution
    if torch.cuda.is_available():
        # Use local rank if launched with torchrun; else default to cuda:0
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda:0"
    else:
        device = "cpu"

    # torchrun --nproc_per_node=NGPUS this_script.py --pattern ...
    pattern = args.pattern.replace("/TODO/", f"/{args.dataset}/")
    run_rank(pattern, local_rank, world_size, device)


if __name__ == "__main__":
    main()