import pyaudio
import wave

# Setup chunk size and format
chunk = 1024
format = pyaudio.paInt16
channels = 1
rate = 16000
record_seconds = 5

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream for recording
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)

print("* recording")

frames = []

# Record for a few seconds
for i in range(0, int(rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)

print("* done recording")

# Stop and close the recording stream
stream.stop_stream()
stream.close()

# Save the recorded data as a WAV file
wf = wave.open('output.wav', 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(format))
wf.setframerate(rate)
wf.writeframes(b''.join(frames))
wf.close()

# Open the WAV file for playback
wf = wave.open('output.wav', 'rb')

# Open stream for playback
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# Read data in chunks and play
data = wf.readframes(chunk)
while data:
    stream.write(data)
    data = wf.readframes(chunk)

# Close the playback stream
stream.stop_stream()
stream.close()

# Terminate the PyAudio object
p.terminate()

print("* playback finished")
