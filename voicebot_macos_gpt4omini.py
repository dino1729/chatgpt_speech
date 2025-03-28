import os
import sounddevice as sd
import soundfile as sf
import tiktoken
import time
import json
import logging
import sys
import random
from config import config
from helper_functions.audio_processors import transcribe_audio_to_text, generate_response, translate_and_speak

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

temperature = config.temperature
max_tokens = config.max_tokens

model_name = "GPT4OMINI"

audio_path = "user_audio.wav"
tts_output_path = "bot_response.mp3"
system_prompt = config.system_prompt
conversation = system_prompt.copy()
encoding = tiktoken.get_encoding("cl100k_base")
max_token_count = 8192
max_timeout = 600
last_activity_time = time.time()

print("Press the Enter key to start/stop recording...")
recording = False
try:
    while True:
        
        # Check if it's time to reset the conversation based on token count or inactivity
        if len(encoding.encode(json.dumps(str(conversation)))) > max_token_count or time.time() - last_activity_time > max_timeout:
            conversation = system_prompt.copy()
            print("Conversation reset")       

        # Update the last activity time
        last_activity_time = time.time()
        # Wait for the Enter key press
        input("Press 'Enter' and ask your question...\n")
        # Toggle the recording state
        recording = not recording

        if recording:
            print("Recording started...")
            recording_data = sd.rec(int(8 * 44100), samplerate=44100, channels=1)
        else:
            print("Recording stopped. Processing audio...")
            sd.stop()
            sf.write(audio_path, recording_data, 44100, 'PCM_16')
            english_text, detected_audio_language = transcribe_audio_to_text(audio_path)
            new_message = {"role": "user", "content": english_text}
            conversation.append(new_message)
            assistant_reply = generate_response(english_text, conversation, model_name, max_tokens, temperature)
            translate_and_speak(assistant_reply, detected_audio_language, tts_output_path, model_name)
            new_assistant_message = {"role": "assistant", "content": assistant_reply}
            conversation.append(new_assistant_message)

            # Delete the audio files
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                if os.path.exists(tts_output_path):
                    os.remove(tts_output_path)
            except Exception as e:
                print("Error deleting audio files:", str(e))
                continue

except KeyboardInterrupt:
    print("\nScript terminated by user.")


