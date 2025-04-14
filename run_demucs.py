import base64
import os
import sys
import tempfile
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pathlib import Path
import subprocess
import json

def download_audio(audio_url, output_path):
    print("Downloading audio...")
    command = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "-o", output_path,
        audio_url
    ]
    subprocess.run(command, check=True)

def encode_base64_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    audio_url = os.environ.get("AUDIO_URL")
    if not audio_url:
        print("No audio URL found.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        output_wav = os.path.join(tmp, "input.%(ext)s")
        download_audio(audio_url, os.path.join(tmp, "input"))

        # Find downloaded file
        for file in os.listdir(tmp):
            if file.endswith(".wav"):
                input_path = os.path.join(tmp, file)
                break
        else:
            print("Downloaded WAV file not found.")
            sys.exit(1)

        print("Loading and separating audio...")
        wav, sr = torchaudio.load(input_path)
        model = get_model(name="htdemucs").cpu()
        sources = apply_model(model, wav.unsqueeze(0), split=True, shifts=1, device="cpu")[0]

        stems = {}
        for name, tensor in zip(model.sources, sources):
            out_path = os.path.join(tmp, f"{name}.wav")
            torchaudio.save(out_path, tensor.cpu(), sr)
            stems[name] = encode_base64_file(out_path)

        with open("output.json", "w") as f:
            json.dump(stems, f)
        print("Separation complete. Stems saved in output.json.")

if __name__ == "__main__":
    main()
