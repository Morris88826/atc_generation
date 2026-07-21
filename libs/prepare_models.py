import sys
sys.path.append('./third_party/AudioSep')
sys.path.append('./third_party/versatile_audio_super_resolution')
import torch
from pathlib import Path
from third_party.AudioSep.pipeline import build_audiosep
from third_party.versatile_audio_super_resolution.inference import Predictor
from third_party.SALT.get_assets import get_assets
from third_party.TokAN.inference import TokANInference
from third_party.F5_TTS.inference import F5TTSInference

def get_models(device):
    checkpoints_dir = Path('./checkpoints/AudioSep')
    models = (
        (
            "https://huggingface.co/spaces/badayvedat/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt",
            checkpoints_dir / "audiosep_base_4M_steps.ckpt"
        ),
        (
            "https://huggingface.co/spaces/badayvedat/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt",
            checkpoints_dir / "music_speech_audioset_epoch_15_esc_89.98.pt"
        )
    )

    model_audiosep = build_audiosep(
        config_yaml='./third_party/AudioSep/config/audiosep_base.yaml',
        checkpoint_path=str(models[0][1]),
        device=torch.device(device),
    )

    model_audio_sr = Predictor()
    model_audio_sr.setup(device=device)

    get_assets(output_dir='./third_party/SALT/assets')
    model_anon = torch.hub.load('BakerBunker/SALT','salt', trust_repo=True, pretrained=True, base=True, device=device)
    speaker_packs = Path('./third_party/SALT/assets').glob('*.pack')
    speaker_packs = [str(pack) for pack in speaker_packs]

    for p in speaker_packs:
        model_anon.add_speaker(p, preprocessed_file=Path(p))

    model_tokan = TokANInference(
        repo_root="./third_party/TokAN",
        cache_dir="./checkpoints/TokAN",
        device=device,
        verbose=False,
    )

    f5tts = F5TTSInference(device=device)

    return {
        "AudioSep": model_audiosep,
        "AudioSR": model_audio_sr,
        "SALT": model_anon,
        "TokAN": model_tokan,
        "F5TTS": f5tts
    }