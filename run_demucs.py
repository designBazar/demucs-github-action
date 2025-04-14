import base64
import sys
import os
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pathlib import Path
import tempfile
import requests

def save_mp3_file(url, out_path):
    # Download the audio file from the URL
    response = requests.get(url)
    with open(out_path, "wb") as f:
        f.write(response.content)

def encode_base64_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    # Get the audio URL from the input
    audio_url = os.environ.get("AUDIO_URL", "")
    if not audio_url:
        print("No audio URL found.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        input_audio = os.path.join(tmp, "input.mp3")
        save_mp3_file(audio_url, input_audio)

        # Convert the audio file to a wav format
        wav, sr = torchaudio.load(input_audio)
        model = get_model(name="htdemucs").cpu()
        sources = apply_model(model, wav.unsqueeze(0), split=True, shifts=1, device="cpu")[0]

        stems = {}
        for name, tensor in zip(model.sources, sources):
            out_path = os.path.join(tmp, f"{name}.wav")
            torchaudio.save(out_path, tensor.cpu(), sr)
            stems[name] = encode_base64_file(out_path)

        with open("output.json", "w") as f:
            import json
            json.dump(stems, f)

if __name__ == "__main__":
    main()
