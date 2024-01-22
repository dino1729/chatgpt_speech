# Python script to test the wakeword detection using porcupine library

import pvporcupine
import pyaudio
import struct
import os
import dotenv

def main():
    # Initialize Porcupine with the 'bumblebee' wake word
    # Get API key from environment variable
    dotenv.load_dotenv()
    porcupine_access_key = os.environ.get("PORCUPINE_ACCESS_KEY")
    porcupine = None
    pa = None
    audio_stream = None
    try:
        porcupine = pvporcupine.create(access_key=porcupine_access_key, keywords=["bumblebee"])

        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )

        print("Press Enter to start detection...")
        input()  # Wait for the user to press Enter

        print("Listening for wake word 'bumblebee'...")

        while True:
            try:
                # Read audio stream with exception_on_overflow set to False
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                result = porcupine.process(pcm)
                if result >= 0:
                    print("Wake word detected")
                    break
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    # Ignore the overflow and continue
                    continue
                else:
                    # An unrelated IOError occurred, print it out and stop the program
                    print(f"An unexpected IOError occurred: {e}")
                    break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if porcupine is not None:
            porcupine.delete()

        if audio_stream is not None:
            audio_stream.close()

        if pa is not None:
            pa.terminate()

if __name__ == '__main__':
    main()
