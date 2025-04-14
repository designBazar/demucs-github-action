import os
import sys
import json
import tempfile
import base64
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pathlib import Path
from urllib.parse import urlparse
import subprocess

def encode_base64_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def download_audio(url, output_path):
    if "youtube.com" in url or "youtu.be" in url:
        cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_path, url]
    else:
        cmd = ["wget", "-O", output_path, url]

    print(f"Downloading audio from: {url}")
    subprocess.run(cmd, check=True)

def convert_mp3_to_wav(mp3_path, wav_path):
    waveform, sample_rate = torchaudio.load(mp3_path)
    torchaudio.save(wav_path, waveform, sample_rate)

def main():
    audio_url = os.environ.get("AUDIO_URL", "")
    if not audio_url:
        print("No AUDIO_URL input provided.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, "input.mp3")
        wav_path = os.path.join(tmp, "input.wav")

        download_audio(audio_url, mp3_path)
        convert_mp3_to_wav(mp3_path, wav_path)

        wav, sr = torchaudio.load(wav_path)
        model = get_model(name="htdemucs").cpu()

        sources = apply_model(model, wav.unsqueeze(0), split=True, shifts=1, device="cpu")[0]

        stems = {}
        for name, tensor in zip(model.sources, sources):
            out_path = os.path.join(tmp, f"{name}.wav")
            torchaudio.save(out_path, tensor.cpu(), sr)
            stems[name] = encode_base64_file(out_path)

        with open("output.json", "w") as f:
            json.dump(stems, f)

if __name__ == "__main__":
    main()
