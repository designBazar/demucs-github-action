import base64
import sys

def encode_audio(file_path):
    with open(file_path, 'rb') as audio_file:
        # Read the audio file and encode it to base64
        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        return encoded_audio

if __name__ == "__main__":
    file_path = sys.argv[1]
    base64_audio = encode_audio(file_path)
    
    # Output the base64 audio to GitHub Actions logs
    print(f"::set-output name=base64_audio::{base64_audio}")
