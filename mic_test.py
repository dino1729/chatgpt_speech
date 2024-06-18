import sounddevice as sd
import soundfile as sf
from helper_functions.audio_processors import transcribe_audio

# Record audio
print("Recording audio...")

duration = 10  # seconds
fs = 44100  # Sample rate
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished

print("Audio recording complete.")
# Save the recorded audio to a file in the current directory
audio_path = "mic_test.wav"
sf.write(audio_path, myrecording, fs, 'PCM_16')

print(f"Audio saved to {audio_path}")

print("Transcribing the audio...")

# Transcribe the audio file and print the text
text, detected_language = transcribe_audio(audio_path)

print(f"Detected language: {detected_language}")
print(f"Transcribed text: {text}")

print("Audio transcription complete.")
