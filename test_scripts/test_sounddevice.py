import sounddevice as sd

# Record audio
duration = 5  # seconds
fs = 16000  # Sample rate
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished

# Play audio
sd.play(myrecording, fs)
sd.wait()  # Wait until playback is finished
