from pathlib import Path
from typing import Optional
import soundfile as sf
import torch
import tqdm
from f5_tts.api import F5TTS


class F5TTSInference:
    def __init__(
        self,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = F5TTS(device=self.device)

    def synthesize(
        self,
        gen_text: str,
        output_wav: str | Path,
        reference_wav: Optional[str | Path],
        reference_text: str,
        duration_reference: Optional[str | Path] = None,
        fix_duration: Optional[float] = None,
        duration_buffer: float = 0.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Path:
        """
        Generate speech and save it to output_wav.

        Parameters
        ----------
        gen_text:
            Text to synthesize.

        output_wav:
            Full output WAV path.

        reference_wav:
            Optional voice-cloning reference. Uses the default reference
            supplied during initialization when omitted.

        reference_text:
            Transcript corresponding to reference_wav.

        duration_reference:
            Optional audio whose duration is used for fix_duration.

        fix_duration:
            Explicit requested output duration in seconds.

        duration_buffer:
            Additional seconds added to the requested duration.

        verbose:
            Whether to print detailed information during synthesis.
        """
        show_info = print if verbose else lambda *args, **kwargs: None

        output_wav = Path(output_wav)
        output_wav.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ref_file = str(reference_wav)
        ref_text = reference_text.strip()

        if fix_duration is not None and duration_reference is not None:
            raise ValueError(
                "Provide either fix_duration or duration_reference, not both."
            )

        if duration_reference is not None:
            info = sf.info(str(duration_reference))
            fix_duration = info.frames / info.samplerate

        if fix_duration is not None:
            fix_duration += duration_buffer

        wav, sr, spec = self.model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=gen_text.strip(),
            file_wave=str(output_wav),
            file_spec=None,
            seed=seed,
            fix_duration=fix_duration,
            show_info=show_info,
            progress = None if not verbose else tqdm
        )

        return output_wav