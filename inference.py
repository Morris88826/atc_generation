import os
import torch
import random
import librosa
import argparse
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from libs.prepare_models import get_models
from libs.apply_aas import mix_speech_and_noise
from third_party.AudioSep.pipeline import separate_audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def log(message, verbose=True):
    if verbose:
        print(message)

def audio_separation(audio_path, model, out_dir, verbose=True):
    log("Running Audio Separation...", verbose)
    noise_out_dir = os.path.join(out_dir, '01_separated_noise')
    speech_out_dir = os.path.join(out_dir, '01_separated_speech')
    combined_out_dir = os.path.join(out_dir, '01_combined_speech')
    os.makedirs(noise_out_dir, exist_ok=True)
    os.makedirs(speech_out_dir, exist_ok=True)
    os.makedirs(combined_out_dir, exist_ok=True)

    # Extract noise
    noise_output_file = os.path.join(noise_out_dir, os.path.basename(audio_path))
    if not os.path.exists(noise_output_file):
        # AudioSep processes the audio at 32 kHz sampling rate
        separate_audio(model, audio_path, 'background noise', noise_output_file, device, verbose=False)

    # Extract speech
    speech_output_file = os.path.join(speech_out_dir, os.path.basename(audio_path))
    if not os.path.exists(speech_output_file):
        separate_audio(model, audio_path, 'human speech', speech_output_file, device, verbose=False)

    # Re-combine the speech and noise
    combined_speech_file = os.path.join(combined_out_dir, os.path.basename(audio_path))
    if not os.path.exists(combined_speech_file):
        clean_audio, sr1 = librosa.load(speech_output_file, sr=None)
        noise_audio, sr2 = librosa.load(noise_output_file, sr=None)
        assert sr1 == sr2
        # Ensure equal lengths
        min_length = min(len(clean_audio), len(noise_audio))
        clean_audio = clean_audio[:min_length]
        noise_audio = noise_audio[:min_length]

        combined_audio = clean_audio + noise_audio

        # Check for clipping
        peak = np.max(np.abs(combined_audio))
        if peak > 1.0:
            combined_audio = combined_audio / peak * 0.99

        wavfile.write(combined_speech_file, sr1, np.round(combined_audio * 32767).astype(np.int16))
    
    log(f"Audio separation completed. Separated files saved in: ", verbose=verbose)
    log(f" - Noise: {noise_out_dir}", verbose=verbose)
    log(f" - Speech: {speech_out_dir}", verbose=verbose)
    log(f" - Combined Speech: {combined_out_dir}", verbose=verbose)
    log("===============================", verbose=verbose)
    return noise_output_file, speech_output_file, combined_speech_file

def audio_super_resolution(audio_path, model, out_dir, verbose=True):
    log("Running Audio Super-Resolution...", verbose=verbose)
    superresolution_out_dir = os.path.join(out_dir, '02_superresolved_speech')
    os.makedirs(superresolution_out_dir, exist_ok=True)
    sr_speech_output_file = os.path.join(superresolution_out_dir, os.path.basename(audio_path))
    if not os.path.exists(sr_speech_output_file):
        model.predict(
            input_file=audio_path,
            output_file=sr_speech_output_file,
            input_cutoff=16000,
            multiband_ensemble=False,
            ddim_steps=50,
            guidance_scale=3.5,
            chunk_size=10.24,
            overlap=0.04,
            seed=0,
            verbose=False,
        )
    
    log(f"Audio super-resolution completed. Super-resolved speech saved in {superresolution_out_dir}.", verbose=verbose)
    log("===============================", verbose=verbose)
    return sr_speech_output_file

def voice_conversion(audio_path, noise_path, model, speaker_packs, out_dir, verbose=True, SEED=42):
    log("Running Voice Conversion...", verbose=verbose)
    random.seed(SEED)  # Set a seed for reproducibility

    vc_out_dir = os.path.join(out_dir, '03_clean_speech_vc')
    os.makedirs(vc_out_dir, exist_ok=True)
    vc_speech_output_file = os.path.join(vc_out_dir, os.path.basename(audio_path))
    if not os.path.exists(vc_speech_output_file):
        voice_converted = model.interpolate(
            audio_path,
            {str(random.choice(speaker_packs)): 1.0},
            topk=4,
            chunksize=5,
            padding=0.5,
        ).cpu().numpy()
        wavfile.write(vc_speech_output_file, 16000, np.round(voice_converted * 32767).astype(np.int16))

    atc_simulated_vc_dir = os.path.join(out_dir, '04_atc_simulated_speech_vc')
    os.makedirs(atc_simulated_vc_dir, exist_ok=True)
    atc_simulated_vc_output_file = os.path.join(atc_simulated_vc_dir, os.path.basename(audio_path))
    if not os.path.exists(atc_simulated_vc_output_file):
        mix_speech_and_noise(
            speech_path=vc_speech_output_file,
            noise_path=noise_path,
            out_path=atc_simulated_vc_output_file,
            sr=16000,
        )

    log(f"Voice conversion completed. Voice-converted speech saved in: ", verbose=verbose)
    log(f" - Voice Converted Speech: {vc_speech_output_file}", verbose=verbose)
    log(f" - ATC Simulated Voice Converted Speech: {atc_simulated_vc_output_file}", verbose=verbose)
    log("===============================", verbose=verbose)

    return vc_speech_output_file, atc_simulated_vc_output_file

def l2_to_l1_accent_conversion(audio_path, noise_path, model, out_dir, verbose=True):
    log("Running L2 to L1 Accent Conversion...", verbose=verbose)
    l2_to_l1_out_dir = os.path.join(out_dir, '03_clean_speech_l2_to_l1')
    os.makedirs(l2_to_l1_out_dir, exist_ok=True)
    l2_to_l1_speech_output_file = os.path.join(l2_to_l1_out_dir, os.path.basename(audio_path))
    if not os.path.exists(l2_to_l1_speech_output_file):
        model.convert(
            source_wav=audio_path,
            output_wav=l2_to_l1_speech_output_file,
            # None means preserve the source speaker
            reference_wav=None,
            # Use TokAN's total-duration-controlled synthesis
            preserve_total_duration=True,
            beam_size=5
        )

    atc_simulated_l2_to_l1_dir = os.path.join(out_dir, '04_atc_simulated_speech_l2_to_l1')
    os.makedirs(atc_simulated_l2_to_l1_dir, exist_ok=True)
    atc_simulated_l2_to_l1_output_file = os.path.join(atc_simulated_l2_to_l1_dir, os.path.basename(audio_path))
    if not os.path.exists(atc_simulated_l2_to_l1_output_file):
        mix_speech_and_noise(
            speech_path=l2_to_l1_speech_output_file,
            noise_path=noise_path,
            out_path=atc_simulated_l2_to_l1_output_file,
            sr=16000,
        )

    log(f"L2 to L1 accent conversion completed. L2 to L1 converted speech saved in: ", verbose=verbose)
    log(f" - L2 to L1 Speech: {l2_to_l1_speech_output_file}", verbose=verbose)
    log(f" - ATC Simulated L2 to L1 Speech: {atc_simulated_l2_to_l1_output_file}", verbose=verbose)
    log("===============================", verbose=verbose)
    
    return l2_to_l1_speech_output_file, atc_simulated_l2_to_l1_output_file

def tts_generation(audio_path, noise_path, transcript, model, out_dir, verbose=True):
    log("Running TTS Generation...", verbose=verbose)
    tts_out_dir = os.path.join(out_dir, '03_clean_speech_tts')
    os.makedirs(tts_out_dir, exist_ok=True)
    tts_speech_output_file = os.path.join(tts_out_dir, os.path.basename(audio_path))
    if not os.path.exists(tts_speech_output_file):
        model.synthesize(
            gen_text=transcript.lower(),
            output_wav=tts_speech_output_file,
            reference_wav=audio_path,
            reference_text=transcript.lower(),
            verbose=False
        )

    atc_simulated_tts_dir = os.path.join(out_dir, '04_atc_simulated_speech_tts')
    os.makedirs(atc_simulated_tts_dir, exist_ok=True)
    atc_simulated_tts_output_file = os.path.join(atc_simulated_tts_dir, os.path.basename(audio_path))
    if not os.path.exists(atc_simulated_tts_output_file):
        mix_speech_and_noise(
            speech_path=tts_speech_output_file,
            noise_path=noise_path,
            out_path=atc_simulated_tts_output_file,
            sr=16000,
        )
        
    log(f"TTS generation completed. TTS generated speech saved in: ", verbose=verbose)
    log(f" - TTS Speech: {tts_speech_output_file}", verbose=verbose)
    log(f" - ATC Simulated TTS Speech: {atc_simulated_tts_output_file}", verbose=verbose)
    log("===============================", verbose=verbose)

    return tts_speech_output_file, atc_simulated_tts_output_file

def main(audio_path, transcript, models, speaker_packs, out_dir, verbose=True):
    log("Running inference on the provided audio file...", verbose=verbose)
    log(f"Audio Path: {audio_path}", verbose=verbose)
    log(f"Transcript: {transcript}", verbose=verbose)
    log("===============================", verbose=verbose)

    # Step 1: Audio Separation
    noise_output_file, speech_output_file, combined_speech_file = audio_separation(audio_path, models['AudioSep'], out_dir, verbose=verbose)

    # Step 2: Audio Super-Resolution
    sr_speech_output_file = audio_super_resolution(speech_output_file, models['AudioSR'], out_dir, verbose=verbose)

    # Step 3-1: Voice Conversion (SALT)
    vc_speech_output_file, atc_simulated_vc_output_file = voice_conversion(sr_speech_output_file, noise_output_file, models['SALT'], speaker_packs, out_dir, verbose=verbose)

    # Step 3-2: L2 to L1 Accent Conversion (TokAN)
    l2_to_l1_speech_output_file, atc_simulated_l2_to_l1_output_file = l2_to_l1_accent_conversion(sr_speech_output_file, noise_output_file, models['TokAN'], out_dir, verbose=verbose)

    # Step 3-3: TTS Generation (F5TTS)
    tts_speech_output_file, atc_simulated_tts_output_file = tts_generation(sr_speech_output_file, noise_output_file, transcript, models['F5TTS'], out_dir, verbose=verbose)

    return {
        "noise_output_file": noise_output_file,
        "speech_output_file": speech_output_file,
        "combined_speech_file": combined_speech_file,
        "sr_speech_output_file": sr_speech_output_file,
        "vc_speech_output_file": vc_speech_output_file,
        "atc_simulated_vc_output_file": atc_simulated_vc_output_file,
        "l2_to_l1_speech_output_file": l2_to_l1_speech_output_file,
        "atc_simulated_l2_to_l1_output_file": atc_simulated_l2_to_l1_output_file,
        "tts_speech_output_file": tts_speech_output_file,
        "atc_simulated_tts_output_file": atc_simulated_tts_output_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare models for inference.")
    parser.add_argument('--audio_path', type=str, required=True, help="Path to the input audio file.")
    parser.add_argument('--transcript', type=str, default=None, help="Transcript of the input audio.")
    parser.add_argument('--out_dir', type=str, default='./output', help="Directory to save the output files.")
    args = parser.parse_args()

    audio_path = args.audio_path
    out_dir = args.out_dir
    if args.transcript is None:
        transcript_path = audio_path.replace('/wav/', '/transcript/').replace('.wav', '.txt')
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found at {transcript_path}. Please provide a transcript using the --transcript argument.")
        with open(transcript_path, 'r') as f:
            transcript = f.read().strip()
    else:
        transcript = args.transcript

    models = get_models(device=device)
    speaker_packs = Path('./third_party/SALT/assets').glob('*.pack')
    speaker_packs = [str(pack) for pack in speaker_packs]

    print("Running inference on the provided audio file...")
    print(f"Audio Path: {audio_path}")
    print(f"Transcript: {transcript}")
    print("===============================")

    main(audio_path, transcript, models, speaker_packs, out_dir, verbose=True)