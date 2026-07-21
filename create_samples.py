import os
import json
import random
import shutil
import argparse
import tqdm


SAMPLES_DIR = "./samples"
ORIGINAL_DIR = os.path.join(SAMPLES_DIR, "00_original_speech")

# Every stage folder inference.py's main() writes into `out_dir` for.
STAGE_DIRS = [
    "01_separated_speech",
    "01_separated_noise",
    "01_combined_speech",
    "02_superresolved_speech",
    "03_clean_speech_vc",
    "04_atc_simulated_speech_vc",
    "03_clean_speech_l2_to_l1",
    "04_atc_simulated_speech_l2_to_l1",
    "03_clean_speech_tts",
    "04_atc_simulated_speech_tts",
]


def label_for(file_id):
    parts = file_id.split('_')
    station = parts[2] if len(parts) > 2 else file_id
    time_raw = parts[-1] if parts else ""
    time_fmt = f"{time_raw[0:2]}:{time_raw[2:4]}" if len(time_raw) == 6 and time_raw.isdigit() else time_raw
    return f"{station} · {time_fmt}"


ALL_DATASETS = ['atco2-asr', 'atc-dataset', 'atcosim']


def find_transcript(file_id):
    for dataset in ALL_DATASETS:
        path = f'./data/{dataset}/data/transcript/{file_id}.txt'
        if os.path.exists(path):
            return open(path).read().strip()
    return ""


def build_manifest():
    stage_dirs = sorted(
        d for d in os.listdir(SAMPLES_DIR)
        if os.path.isdir(os.path.join(SAMPLES_DIR, d)) and d != "transcripts"
    )
    id_sets = [
        {f[:-4] for f in os.listdir(os.path.join(SAMPLES_DIR, d)) if f.endswith('.wav')}
        for d in stage_dirs
    ]
    common_ids = sorted(set.intersection(*id_sets)) if id_sets else []

    manifest = [
        {"id": file_id, "label": label_for(file_id), "transcript": find_transcript(file_id)}
        for file_id in common_ids
    ]

    manifest_path = os.path.join(SAMPLES_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} entries to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy already-generated pipeline outputs from data/<dataset>/aug/ into samples/ for the demo page.")
    parser.add_argument('--dataset', type=str, default='atco2-asr', choices=['atc-dataset', 'atco2-asr', 'atcosim'], help="Dataset to draw utterances from (must already have been processed via main.py).")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of utterances to copy.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed used when picking utterances.")
    args = parser.parse_args()

    for stage_dir in [ORIGINAL_DIR] + [os.path.join(SAMPLES_DIR, d) for d in STAGE_DIRS]:
        os.makedirs(stage_dir, exist_ok=True)

    aug_dir = f'./data/{args.dataset}/aug'
    wav_dir = f'./data/{args.dataset}/data/wav'

    # Only utterances that have made it through every stage are usable.
    id_sets = [
        {os.path.splitext(f)[0] for f in os.listdir(os.path.join(aug_dir, stage)) if f.endswith('.wav')}
        for stage in STAGE_DIRS
    ]
    complete_ids = set.intersection(*id_sets) if id_sets else set()
    complete_ids &= {os.path.splitext(f)[0] for f in os.listdir(wav_dir) if f.endswith('.wav')}
    if not complete_ids:
        raise FileNotFoundError(f"No utterances in '{args.dataset}' have outputs for every stage under {aug_dir}. Run main.py first.")

    file_ids = random.Random(args.seed).sample(sorted(complete_ids), min(args.num_samples, len(complete_ids)))
    print(f"Copying {len(file_ids)} of {len(complete_ids)} fully-processed utterance(s) from '{args.dataset}'.")

    for file_id in tqdm.tqdm(file_ids):
        shutil.copy(os.path.join(wav_dir, f'{file_id}.wav'), os.path.join(ORIGINAL_DIR, f'{file_id}.wav'))
        for stage in STAGE_DIRS:
            shutil.copy(os.path.join(aug_dir, stage, f'{file_id}.wav'), os.path.join(SAMPLES_DIR, stage, f'{file_id}.wav'))

    build_manifest()
