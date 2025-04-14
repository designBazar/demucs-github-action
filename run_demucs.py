import os
import requests
import base64
import tempfile
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile
from torchaudio import save

def download_audio(url, path):
    print(f"Downloading from {url}")
    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)

def encode_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def run():
    url = os.getenv("AUDIO_URL")
    if not url:
        raise Exception("Missing AUDIO_URL")

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "audio.mp3")
        out_dir = os.path.join(tmp, "stems")
        os.makedirs(out_dir, exist_ok=True)

        download_audio(url, input_path)

        print("Running Demucs...")
        model = get_model(name="htdemucs")
        wav = AudioFile(input_path).read(streams=0, channels=1)[0]
        out = apply_model(model, wav[None])[0]

        for name, source in zip(model.sources, out):
            out_path = os.path.join(out_dir, f"{name}.mp3")
            save(out_path, source.unsqueeze(0), 44100, format="mp3")
            print(f"--- {name.upper()} BASE64 START ---")
            print(encode_base64(out_path))
            print(f"--- {name.upper()} BASE64 END ---")

if __name__ == "__main__":
    run()
