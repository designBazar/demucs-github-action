import os
import subprocess
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
import tempfile
import json
import requests

def download_audio(audio_url, output_path):
    # Download the MP3 file directly using requests
    response = requests.get(audio_url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def main():
    # Get the audio URL from the environment variable
    audio_url = os.getenv("AUDIO_URL")
    if not audio_url:
        print("No audio URL provided.")
        exit(1)

    # Temporary directory to store audio and output
    with tempfile.TemporaryDirectory() as tmp:
        input_audio_path = os.path.join(tmp, "input.mp3")
        download_audio(audio_url, input_audio_path)

        # Load the audio file using torchaudio
        wav, sr = torchaudio.load(input_audio_path)

        # Load the Demucs model
        model = get_model(name="htdemucs").cpu()
        sources = apply_model(model, wav.unsqueeze(0), split=True, shifts=1, device="cpu")[0]

        # Save each separated stem (audio source) as a separate file
        stems = {}
        for name, tensor in zip(model.sources, sources):
            out_path = os.path.join(tmp, f"{name}.wav")
            torchaudio.save(out_path, tensor.cpu(), sr)
            stems[name] = out_path

        # Output results as JSON (base64 could be added here if necessary)
        with open("/tmp/output.json", "w") as f:
            json.dump(stems, f)

if __name__ == "__main__":
    main()
