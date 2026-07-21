from pathlib import Path
from urllib.request import urlretrieve
import zipfile


def get_assets(output_dir):
    """
    Download and extract the SALT assets from the provided URL.
    """
    url = "https://github.com/BakerBunker/SALT/releases/download/1.0.0/librispeech-pack.zip"
    output_dir = Path(output_dir)
    zip_path = output_dir / "librispeech-pack.zip"

    # Create the destination folder
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        # Download
        print(f"Downloading to {zip_path}...")
        urlretrieve(url, zip_path)

        # Extract
        print(f"Extracting to {output_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(output_dir)

        print("Done.")