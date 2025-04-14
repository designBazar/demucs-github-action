import os
import tempfile
import json
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
import requests

def download_audio(audio_url, output_path):
    response = requests.get(audio_url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def main():
    audio_url = os.getenv("AUDIO_URL")
    if not audio_url:
        print("No audio URL provided.")
        exit(1)

    # Directory for output in the repository
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Temporary directory to store audio
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
            out_path = os.path.join(output_dir, f"{name}.wav")
            torchaudio.save(out_path, tensor.cpu(), sr)
            stems[name] = out_path

        # Output results as JSON
        output_json_path = os.path.join(output_dir, "output.json")
        with open(output_json_path, "w") as f:
            json.dump(stems, f)

        print(f"Output saved to: {output_json_path}")

if __name__ == "__main__":
    main()
