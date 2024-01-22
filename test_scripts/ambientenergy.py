import numpy as np
import pyaudio

# Initialize PyAudio and start the stream
pa = pyaudio.PyAudio()
stream = pa.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=1024)

# Record ambient noise
frames = []
for _ in range(0, int(16000 / 1024 * 10)):  # Record for 10 seconds
    data = stream.read(1024)
    frames.append(np.frombuffer(data, dtype=np.int16))

# Calculate the average energy of the ambient noise
ambient_energy = np.mean([np.sum(np.square(frame)) for frame in frames])

threshold = np.all(np.abs(np.frombuffer(data, dtype=np.int16)))

# Close the stream
stream.stop_stream()
stream.close()
pa.terminate()

print("Average ambient energy:", ambient_energy)
print("Threshold:", threshold)
