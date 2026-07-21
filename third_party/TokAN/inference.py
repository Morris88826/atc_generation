"""Reusable TokAN inference API.

This module wraps the original CLI-style inference script in a class that:
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from hyperpyyaml import load_hyperpyyaml

PathLike = Union[str, os.PathLike]
DEFAULT_CONFIG_FILENAME = "tokan.model-only.yaml"


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return torch.device("cuda")
    if device != "auto":
        raise ValueError("device must be one of: 'auto', 'cpu', or 'cuda'.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _deduplicate(tokens: list[int]) -> tuple[list[int], list[int]]:
    """Run-length encode token IDs into unique tokens and repeat counts."""
    if not tokens:
        return [], []

    unique_tokens: list[int] = []
    durations: list[int] = []
    previous = tokens[0]
    count = 1

    for token in tokens[1:]:
        if token == previous:
            count += 1
        else:
            unique_tokens.append(previous)
            durations.append(count)
            previous = token
            count = 1

    unique_tokens.append(previous)
    durations.append(count)
    return unique_tokens, durations


class TokANInference:
    """Load TokAN once and reuse it for wav-to-wav inference."""

    def __init__(
        self,
        *,
        repo_root: Optional[PathLike] = None,
        model_config: Optional[PathLike] = None,
        lm_checkpoint: Optional[PathLike] = None,
        quantizer_checkpoint: Optional[PathLike] = None,
        hift_checkpoint: Optional[PathLike] = None,
        hf_repo_id: str = "Piping/TokAN",
        hf_subfolder: str = "checkpoints",
        model_tag: str = "default",
        lm_filename: str = "arlm.pt",
        quantizer_filename: str = "quantizer.pt",
        hift_filename: str = "hift.pt",
        config_filename: str = DEFAULT_CONFIG_FILENAME,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        cache_dir: PathLike = "checkpoints",
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.device = _resolve_device(device)
        self.cache_dir = str(Path(cache_dir).expanduser())

        # HyperPyYAML configs may reference Python classes inside the TokAN repo.
        # Add that repository to sys.path when this wrapper is used elsewhere.
        if repo_root is not None:
            repo_root = Path(repo_root).expanduser().resolve()
            if not repo_root.is_dir():
                raise FileNotFoundError(f"TokAN repo not found: {repo_root}")
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

        self._log(f"Using device: {self.device}")

        config_path = self._resolve_file(
            model_config,
            repo_id=hf_repo_id,
            subfolder=hf_subfolder,
            model_tag=model_tag,
            filename=config_filename,
            revision=hf_revision,
            token=hf_token,
        )
        lm_path = self._resolve_file(
            lm_checkpoint,
            repo_id=hf_repo_id,
            subfolder=hf_subfolder,
            model_tag=model_tag,
            filename=lm_filename,
            revision=hf_revision,
            token=hf_token,
        )
        quantizer_path = self._resolve_file(
            quantizer_checkpoint,
            repo_id=hf_repo_id,
            subfolder=hf_subfolder,
            model_tag=model_tag,
            filename=quantizer_filename,
            revision=hf_revision,
            token=hf_token,
        )
        hift_path = self._resolve_file(
            hift_checkpoint,
            repo_id=hf_repo_id,
            subfolder=hf_subfolder,
            model_tag=model_tag,
            filename=hift_filename,
            revision=hf_revision,
            token=hf_token,
        )

        self.lm, self.quantizer, self.hift, self.sample_rate = self._load_models(
            config_path=config_path,
            lm_checkpoint=lm_path,
            quantizer_checkpoint=quantizer_path,
            hift_checkpoint=hift_path,
        )

        # Load the speaker encoder once instead of rebuilding it per utterance.
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
        except Exception as exc:
            raise RuntimeError(
                "Failed to import resemblyzer. Install it with: pip install resemblyzer"
            ) from exc

        encoder_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.speaker_encoder = VoiceEncoder(device=encoder_device)
        self.preprocess_wav = preprocess_wav
        self._log("TokAN models loaded.")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[TokAN] {message}")

    def _download(
        self,
        *,
        repo_id: str,
        subfolder: str,
        model_tag: str,
        filename: str,
        revision: Optional[str],
        token: Optional[str],
    ) -> str:
        bundle_subfolder = f"{subfolder}/{model_tag}" if subfolder else model_tag
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=bundle_subfolder,
            local_dir=self.cache_dir,
            revision=revision,
            token=token,
        )

    def _resolve_file(
        self,
        local_path: Optional[PathLike],
        *,
        repo_id: str,
        subfolder: str,
        model_tag: str,
        filename: str,
        revision: Optional[str],
        token: Optional[str],
    ) -> str:
        if local_path is not None:
            path = Path(local_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            return str(path)

        return self._download(
            repo_id=repo_id,
            subfolder=subfolder,
            model_tag=model_tag,
            filename=filename,
            revision=revision,
            token=token,
        )

    def _load_models(
        self,
        *,
        config_path: str,
        lm_checkpoint: str,
        quantizer_checkpoint: str,
        hift_checkpoint: str,
    ) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, int]:
        with open(config_path, "r", encoding="utf-8") as file:
            configs = load_hyperpyyaml(file)

        lm = configs["lm"].to(self.device).eval()
        lm_state = torch.load(lm_checkpoint, map_location="cpu")
        for key in ("epoch", "step"):
            lm_state.pop(key, None)
        lm.load_state_dict(lm_state, strict=True)

        quantizer = configs["quantizer"].to(self.device).eval()
        quantizer_state = torch.load(quantizer_checkpoint, map_location="cpu")
        for key in ("epoch", "step"):
            quantizer_state.pop(key, None)
        quantizer.load_state_dict(quantizer_state, strict=True)

        hift = configs["hift"].to(self.device).eval()
        hift.load_state_dict(
            torch.load(hift_checkpoint, map_location="cpu"),
            strict=True,
        )

        for model in (lm, quantizer, hift):
            for parameter in model.parameters():
                parameter.requires_grad = False

        return lm, quantizer, hift, int(configs["sample_rate"])

    @staticmethod
    def _load_audio(path: PathLike, target_sr: int) -> torch.Tensor:
        path = str(Path(path).expanduser())
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=target_sr,
            )
        return waveform.squeeze(0)

    def _speaker_embedding(self, reference_wav: PathLike) -> torch.Tensor:
        preprocessed = self.preprocess_wav(str(reference_wav))
        embedding = self.speaker_encoder.embed_utterance(preprocessed)
        embedding = torch.from_numpy(
            np.asarray(embedding, dtype=np.float32)
        ).unsqueeze(0)
        return F.normalize(embedding, p=2, dim=-1).to(self.device)

    @torch.inference_mode()
    def generate(
        self,
        source_wav: PathLike,
        *,
        reference_wav: Optional[PathLike] = None,
        beam_size: int = 10,
        n_timesteps: int = 32,
        full_cfg: float = 0.0,
        cond_cfg: float = 1.0,
        spk_cfg: float = 1.0,
        preserve_total_duration: bool = False,
        use_lm_duration: bool = False,
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """Generate audio and return ``(waveform, sample_rate, mel)``."""
        source_wav = Path(source_wav).expanduser().resolve()
        if not source_wav.is_file():
            raise FileNotFoundError(f"Source wav not found: {source_wav}")

        if reference_wav is None:
            reference_wav = source_wav
        reference_wav = Path(reference_wav).expanduser().resolve()
        if not reference_wav.is_file():
            raise FileNotFoundError(f"Reference wav not found: {reference_wav}")

        source = self._load_audio(source_wav, target_sr=16000).to(self.device)
        source = source.unsqueeze(0)
        source_length = torch.tensor(
            [source.shape[1]],
            dtype=torch.long,
            device=self.device,
        )

        _, source_tokens, source_token_length = self.quantizer.quantize(
            source,
            source_length,
        )
        self._log(f"Extracted {source_token_length.item()} source tokens.")

        speaker_embedding = self._speaker_embedding(reference_wav)

        target_token_list = [
            int(token)
            for token in self.lm.inference(
                src_tokens=source_tokens,
                src_token_len=source_token_length,
                beam_size=beam_size,
            )
        ]
        if not target_token_list:
            raise RuntimeError("The TokAN language model produced no target tokens.")

        self._log(f"Generated {len(target_token_list)} target tokens.")

        if self.quantizer.flow.duration_predictor is not None:
            deduplicated_tokens, lm_durations = _deduplicate(target_token_list)
            target_tokens = torch.tensor(
                deduplicated_tokens,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            target_token_length = torch.tensor(
                [target_tokens.shape[1]],
                dtype=torch.long,
                device=self.device,
            )
            duration = (
                torch.tensor(lm_durations, dtype=torch.long, device=self.device).unsqueeze(0)
                if use_lm_duration
                else None
            )
        else:
            target_tokens = torch.tensor(
                target_token_list,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            target_token_length = torch.tensor(
                [target_tokens.shape[1]],
                dtype=torch.long,
                device=self.device,
            )
            duration = torch.ones_like(target_tokens)

        if preserve_total_duration:
            total_duration = source_token_length.long()
            duration = None
        else:
            total_duration = None

        mel, _ = self.quantizer.synthesize(
            quant_indices=target_tokens,
            feature_lengths=target_token_length,
            spk_embed=speaker_embedding,
            full_cfg=float(full_cfg),
            cond_cfg=float(cond_cfg),
            spk_cfg=float(spk_cfg),
            n_timesteps=int(n_timesteps),
            total_duration=total_duration,
            use_source_duration=(duration is not None),
        )

        mel_for_hift = mel.transpose(1, 2)
        cache_source = torch.zeros(1, 1, 0, device=self.device)
        speech, _ = self.hift.inference(
            speech_feat=mel_for_hift,
            cache_source=cache_source,
        )

        waveform = speech.detach().cpu().squeeze().numpy().astype(np.float32)
        return waveform, self.sample_rate, mel.detach().cpu()

    def convert(
        self,
        source_wav: PathLike,
        output_wav: PathLike,
        *,
        reference_wav: Optional[PathLike] = None,
        mel_path: Optional[PathLike] = None,
        beam_size: int = 10,
        n_timesteps: int = 32,
        full_cfg: float = 0.0,
        cond_cfg: float = 1.0,
        spk_cfg: float = 1.0,
        preserve_total_duration: bool = False,
        use_lm_duration: bool = False,
    ) -> Path:
        """Run TokAN and save the converted waveform to ``output_wav``."""
        waveform, sample_rate, mel = self.generate(
            source_wav=source_wav,
            reference_wav=reference_wav,
            beam_size=beam_size,
            n_timesteps=n_timesteps,
            full_cfg=full_cfg,
            cond_cfg=cond_cfg,
            spk_cfg=spk_cfg,
            preserve_total_duration=preserve_total_duration,
            use_lm_duration=use_lm_duration,
        )

        output_wav = Path(output_wav).expanduser().resolve()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, sample_rate)
        self._log(f"Saved audio: {output_wav}")

        if mel_path is not None:
            mel_path = Path(mel_path).expanduser().resolve()
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mel, mel_path)
            self._log(f"Saved Mel tensor: {mel_path}")

        return output_wav

    def close(self) -> None:
        """Optional explicit cleanup when the model is no longer needed."""
        for name in ("lm", "quantizer", "hift", "speaker_encoder"):
            if hasattr(self, name):
                delattr(self, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()