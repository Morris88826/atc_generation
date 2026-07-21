import os
import glob
import torch
import tqdm
import argparse
from pathlib import Path
from libs.prepare_models import get_models
from inference import main

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare models for inference.")
    parser.add_argument('--dataset', type=str, required=True, choices=['atc-dataset', 'atco2-asr', 'atcosim'], help="Dataset to use for inference.")
    parser.add_argument('--out_dir', type=str, default='./data', help="Directory to save the output files.")
    args = parser.parse_args()

    dataset_name = args.dataset
    out_dir = os.path.join(args.out_dir, args.dataset, 'aug')
    os.makedirs(out_dir, exist_ok=True)

    models = get_models(device=device)
    speaker_packs = Path('./third_party/SALT/assets').glob('*.pack')
    speaker_packs = [str(pack) for pack in speaker_packs]

    audio_paths = sorted(glob.glob(f'./data/{dataset_name}/data/wav/*.wav'))
    print(f"Found {len(audio_paths)} audio files in the dataset '{dataset_name}'.")

    for audio_path in tqdm.tqdm(audio_paths):
        transcript_path = audio_path.replace('/wav/', '/transcript/').replace('.wav', '.txt')
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found at {transcript_path}.")
        with open(transcript_path, 'r') as f:
            transcript = f.read().strip()

        main(audio_path, transcript, models, speaker_packs, out_dir, verbose=False)