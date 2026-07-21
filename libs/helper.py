from pathlib import Path
from IPython.display import Audio, HTML, display


def show_audio_grid(column_names, audios):
    """
    Display audio samples in a one-row grid.

    Parameters
    ----------
    column_names : list[str]
        Names shown above each audio player.

    audios : list
        Each item can be:
        - a file path: "audio.wav"
        - a Path object
        - a tuple: (audio_array, sample_rate)
    """
    if len(column_names) != len(audios):
        raise ValueError(
            f"Expected the same number of names and audios, "
            f"got {len(column_names)} names and {len(audios)} audios."
        )

    header_cells = []
    audio_cells = []

    for name, audio in zip(column_names, audios):
        if isinstance(audio, (str, Path)):
            player = Audio(filename=str(audio))
        elif isinstance(audio, tuple) and len(audio) == 2:
            audio_array, sample_rate = audio
            player = Audio(audio_array, rate=sample_rate)
        else:
            raise TypeError(
                "Each audio must be a file path or an "
                "(audio_array, sample_rate) tuple."
            )

        header_cells.append(f"<th>{name}</th>")
        audio_cells.append(f"<td>{player._repr_html_()}</td>")

    html = f"""
    <style>
        .audio-grid {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}

        .audio-grid th {{
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }}

        .audio-grid td {{
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
        }}

        .audio-grid audio {{
            width: 100%;
        }}
    </style>

    <table class="audio-grid">
        <tr>
            {''.join(header_cells)}
        </tr>
        <tr>
            {''.join(audio_cells)}
        </tr>
    </table>
    """

    display(HTML(html))