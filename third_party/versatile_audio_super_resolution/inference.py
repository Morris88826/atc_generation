import gc
import os
import random
import numpy as np
from scipy.signal.windows import hann
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path
import tempfile
import argparse
import librosa
from audiosr import build_model, super_resolution
from scipy import signal
import pyloudnorm as pyln


import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
    if (len(array_1.shape) == 1) & (len(array_2.shape) == 1):
        if array_1.shape[0] > array_2.shape[0]:
            array_1 = array_1[:array_2.shape[0]]
        elif array_1.shape[0] < array_2.shape[0]:
            array_1 = np.pad(array_1, ((array_2.shape[0] - array_1.shape[0], 0)), 'constant', constant_values=0)
    else:
        if array_1.shape[1] > array_2.shape[1]:
            array_1 = array_1[:,:array_2.shape[1]]
        elif array_1.shape[1] < array_2.shape[1]:
            padding = array_2.shape[1] - array_1.shape[1]
            array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    return array_1


def lr_filter(audio, cutoff, filter_type, order=12, sr=48000):
    audio = audio.T
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio.T

class Predictor(BasePredictor):
    def setup(self, model_name="basic", device="auto"):
        self.output_sr = 48000
        self.audiosr = build_model(
            model_name=model_name,
            device=device,
        )

    @staticmethod
    def _log(message, verbose):
        if verbose:
            print(message)

    def process_audio(
        self,
        input_file,
        input_cutoff=16000,
        multiband_ensemble=False,
        chunk_size=10.24,
        overlap=0.04,
        seed=None,
        guidance_scale=3.5,
        ddim_steps=50,
        verbose=False,
    ):
        input_file = str(input_file)
        processing_sr = input_cutoff * 2

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between 0 and 1.")

        # Exact target length based on the original file.
        info = sf.info(input_file)
        original_duration = info.frames / info.samplerate
        target_length = round(original_duration * self.output_sr)

        audio, _ = librosa.load(
            input_file,
            sr=processing_sr,
            mono=False,
        )

        # Convert librosa's [channels, samples] to [samples, channels].
        if audio.ndim == 2:
            audio = audio.T
            channels = [audio[:, i] for i in range(audio.shape[1])]
            is_stereo = True
        else:
            channels = [audio]
            is_stereo = False

        chunk_samples = round(chunk_size * processing_sr)
        overlap_samples = round(overlap * chunk_samples)
        hop_samples = chunk_samples - overlap_samples

        output_overlap = round(
            overlap_samples * self.output_sr / processing_sr
        )

        self._log(f"Input: {input_file}", verbose)
        self._log(f"Original duration: {original_duration:.3f}s", verbose)
        self._log(f"Processing rate: {processing_sr} Hz", verbose)
        self._log(f"Chunk size: {chunk_size}s", verbose)

        reconstructed_channels = []

        for channel_index, channel in enumerate(channels):
            output = np.zeros(target_length, dtype=np.float32)

            starts = list(range(0, len(channel), hop_samples))

            for chunk_index, start in enumerate(starts):
                end = min(start + chunk_samples, len(channel))
                chunk = channel[start:end]
                original_chunk_length = len(chunk)

                # Pad short chunks only for AudioSR inference.
                if original_chunk_length < chunk_samples:
                    chunk = np.pad(
                        chunk,
                        (0, chunk_samples - original_chunk_length),
                    )

                self._log(
                    f"Channel {channel_index + 1}, "
                    f"chunk {chunk_index + 1}/{len(starts)}",
                    verbose,
                )

                # AudioSR expects a file path.
                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=True,
                ) as temp_wav:
                    sf.write(temp_wav.name, chunk, processing_sr)

                    out_chunk = super_resolution(
                        self.audiosr,
                        temp_wav.name,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        ddim_steps=ddim_steps,
                        latent_t_per_second=12.8,
                    )

                out_chunk = np.squeeze(np.asarray(out_chunk))

                # Remove output corresponding to zero padding.
                keep_samples = round(
                    original_chunk_length
                    * self.output_sr
                    / processing_sr
                )
                out_chunk = out_chunk[:keep_samples].astype(np.float32)

                # Loudness matching can fail for very short/silent audio.
                try:
                    input_meter = pyln.Meter(processing_sr)
                    output_meter = pyln.Meter(self.output_sr)

                    input_loudness = input_meter.integrated_loudness(
                        chunk[:original_chunk_length]
                    )
                    output_loudness = output_meter.integrated_loudness(
                        out_chunk
                    )

                    if (
                        np.isfinite(input_loudness)
                        and np.isfinite(output_loudness)
                    ):
                        out_chunk = pyln.normalize.loudness(
                            out_chunk,
                            output_loudness,
                            input_loudness,
                        )
                except ValueError:
                    self._log("Skipped loudness matching.", verbose)

                # Crossfade only when there is more than one chunk.
                if len(starts) > 1 and output_overlap > 0:
                    actual_overlap = min(output_overlap, len(out_chunk))

                    if chunk_index > 0:
                        out_chunk[:actual_overlap] *= np.linspace(
                            0,
                            1,
                            actual_overlap,
                        )

                    if chunk_index < len(starts) - 1:
                        out_chunk[-actual_overlap:] *= np.linspace(
                            1,
                            0,
                            actual_overlap,
                        )

                output_start = round(
                    start * self.output_sr / processing_sr
                )
                output_end = min(
                    output_start + len(out_chunk),
                    target_length,
                )

                length = output_end - output_start

                if length > 0:
                    output[output_start:output_end] += out_chunk[:length]

            reconstructed_channels.append(output)

        if is_stereo:
            reconstructed_audio = np.stack(
                reconstructed_channels,
                axis=1,
            )
        else:
            reconstructed_audio = reconstructed_channels[0]

        if multiband_ensemble:
            crossover_freq = input_cutoff - 1000

            original_48k, _ = librosa.load(
                input_file,
                sr=self.output_sr,
                mono=False,
            )

            if original_48k.ndim == 2:
                original_48k = original_48k.T

            original_48k = original_48k[:target_length]
            reconstructed_audio = reconstructed_audio[:target_length]

            low = lr_filter(
                original_48k,
                crossover_freq,
                "lowpass",
                order=10,
                sr=self.output_sr,
            )

            high = lr_filter(
                reconstructed_audio,
                crossover_freq,
                "highpass",
                order=10,
                sr=self.output_sr,
            )

            high = lr_filter(
                high,
                23000,
                "lowpass",
                order=2,
                sr=self.output_sr,
            )

            reconstructed_audio = low + high

        reconstructed_audio = np.nan_to_num(
            reconstructed_audio
        ).astype(np.float32)

        peak = np.max(np.abs(reconstructed_audio))

        if peak > 1.0:
            reconstructed_audio = reconstructed_audio / peak * 0.99

        return reconstructed_audio

    def predict(
        self,
        input_file: Path = Input(
            description="Input audio file",
        ),
        output_file: str = Input(
            description="Full output WAV path",
        ),
        input_cutoff: int = Input(
            description="Use 16000 for 32-kHz input",
            default=16000,
        ),
        multiband_ensemble: bool = Input(
            description="Use original low band with generated high band",
            default=False,
        ),
        ddim_steps: int = Input(
            description="DDIM inference steps",
            default=50,
        ),
        guidance_scale: float = Input(
            description="Guidance scale",
            default=3.5,
        ),
        overlap: float = Input(
            description="Chunk overlap fraction",
            default=0.04,
        ),
        chunk_size: float = Input(
            description="Chunk duration in seconds",
            default=10.24,
        ),
        seed: int = Input(
            description="Random seed; 0 means random",
            default=0,
        ),
        verbose: bool = Input(
            description="Print processing messages",
            default=False,
        ),
    ) -> Path:
        if seed == 0:
            seed = random.randint(0, 2**32 - 1)

        waveform = self.process_audio(
            input_file=input_file,
            input_cutoff=input_cutoff,
            multiband_ensemble=multiband_ensemble,
            chunk_size=chunk_size,
            overlap=overlap,
            seed=seed,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            verbose=verbose,
        )

        output_file = os.path.abspath(output_file)
        output_dir = os.path.dirname(output_file)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        sf.write(
            output_file,
            waveform,
            self.output_sr,
            subtype="PCM_16",
        )

        self._log(f"Saved: {output_file}", verbose)

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return Path(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upsample audio using AudioSR."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Full path of the output WAV file.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM inference steps.",
    )
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=10.24,
        help="Chunk duration in seconds.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Use 0 to generate a random seed.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.04,
        help="Chunk overlap ratio. For example, 0.04 means 4%%.",
    )
    parser.add_argument(
        "--multiband_ensemble",
        action="store_true",
        help="Preserve the original low-frequency band.",
    )
    parser.add_argument(
        "--input_cutoff",
        type=int,
        default=16000,
        help="Input cutoff frequency in Hz. Use 16000 for 32-kHz audio.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print processing information.",
    )

    args = parser.parse_args()

    predictor = Predictor()
    predictor.setup(device="auto")

    output_path = predictor.predict(
        input_file=args.input,
        output_file=args.output,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        input_cutoff=args.input_cutoff,
        multiband_ensemble=args.multiband_ensemble,
        verbose=args.verbose,
    )

    del predictor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
