import base64
import sys
import os
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pathlib import Path
import tempfile

def save_base64_file(b64_string, out_path):
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_string))

def encode_base64_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    audio_b64 = os.environ.get("INPUT_BASE64", "")
    if not audio_b64:
        print("No audio input found.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        input_wav = os.path.join(tmp, "input.wav")
        save_base64_file(audio_b64, input_wav)

        wav, sr = torchaudio.load(input_wav)
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
