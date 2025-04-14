import base64

# Read the processed audio file path
with open("/tmp/processed_audio_path.txt", "r") as f:
    processed_audio_path = f.read().strip()

# Open the processed audio file and encode it to base64
with open(processed_audio_path, "rb") as audio_file:
    audio_data = base64.b64encode(audio_file.read()).decode('utf-8')

# Save the Base64 encoded audio data
with open("/tmp/processed_audio_base64.txt", "w") as f:
    f.write(audio_data)
