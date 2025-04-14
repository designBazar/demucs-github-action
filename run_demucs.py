import os
import requests
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
import tempfile

def download_audio(audio_url, output_path):
    # Download the MP3 file directly using requests
    response = requests.get(audio_url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def main():
    audio_url = os.getenv("AUDIO_URL")
    if not audio_url:
        print("No audio URL provided.")
        exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        input_audio_path = os.path.join(tmp, "input.mp3")
        download_audio(audio_url, input_audio_path)

        # Load the audio file using torchaudio
        wav, sr = torchaudio.load(input_audio_path)

        # Load the Demucs model
        model = get_model(name="htdemucs").cpu()
        sources = apply_model(model, wav.unsqueeze(0), split=True, shifts=1, device="cpu")[0]

        # Save the processed audio (e.g., vocals) as a .wav file
        processed_audio_path = os.path.join(tmp, "processed_audio.wav")
        torchaudio.save(processed_audio_path, sources[0].cpu(), sr)

        # Save the processed audio file path
        with open("/tmp/processed_audio_path.txt", "w") as f:
            f.write(processed_audio_path)

if __name__ == "__main__":
    main()
