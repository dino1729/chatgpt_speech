import os
import sounddevice as sd
import soundfile as sf
import tiktoken
import time
import json
import logging
import sys
from config import config
from helper_functions.audio_processors import transcribe_audio_to_text, generate_response, translate_and_speak
import RPi.GPIO as GPIO
from aiy.leds import (Leds, Pattern, PrivacyLed, RgbLeds, Color)
import random

def update_led(led_state, color=Color.WHITE, brightness=1.0):
    # Ensure brightness is within the allowed range [0.0, 1.0]
    brightness = max(0.0, min(brightness, 1.0))

    # Scale the color values based on the brightness level
    scaled_color = tuple(int(c * brightness) for c in color)

    if led_state == 'ON':
        leds.update(Leds.rgb_on(scaled_color))
    elif led_state == 'OFF':
        leds.update(Leds.rgb_off())
    elif led_state == 'BLINK':
        leds.pattern = Pattern.blink(100)
        leds.update(Leds.rgb_pattern(scaled_color))
    elif led_state == 'BREATHE':
        leds.pattern = Pattern.breathe(1000)
        leds.update(Leds.rgb_pattern(scaled_color))

'''
This script transcribes the native audio file to english language, sends this english text to GPT-3 for completion, and then translates the completed english text back to the native language and generates the audio response.
'''
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

temperature = config.temperature
max_tokens = config.max_tokens

# Randomly select a model_index value from the list of model names
model_names = ["GROQ_MIXTRAL", "GROQ_LLAMA", "GPT35TURBO", "GPT4", "GEMINI", "COHERE"]
model_index = random.randint(0, len(model_names) - 1)
model_name = model_names[model_index]
print("Starting with model:", model_name)

system_prompt = config.system_prompt
audio_path = "user_audio.wav"
tts_output_path = "bot_response.mp3"

encoding = tiktoken.get_encoding("cl100k_base")
max_token_count = 8192
max_timeout = 600
last_activity_time = time.time()

# Set the initial conversation to the default system prompt
conversation = system_prompt.copy()

recording = False
prompt_printed = False  # Flag to control the printing of the prompt

# Set up GPIO
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 23
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Define constants for the double press detection
DOUBLE_PRESS_MAX_DELAY = 0.5  # Maximum delay between presses in seconds

# Initialize state for double press detection
last_press_time = 0
double_press_detected = False

# Initialize the LEDs
leds = Leds()
update_led('ON', Color.CYAN, 0.1)  # Solid CYAN while idle

# Main loop
try:
    while True:
        
        # Check if it's time to reset the conversation based on token count or inactivity
        if len(encoding.encode(json.dumps(str(conversation)))) > max_token_count or time.time() - last_activity_time > max_timeout:
            conversation = system_prompt.copy()  # Reset the conversation to the default
            print("Conversation reset. Changing Model...") 
            # Increment the model index
            model_index = (model_index + 1) % len(model_names)
            # Get the current model name
            model_name = model_names[model_index]
            print("Swapped to model:", model_name)
            prompt_printed = False  # Allow prompt to be printed again after model swap

        # Update the last activity time
        last_activity_time = time.time()

        # Print the prompt only once at the start or after processing is done
        if not recording and not prompt_printed:
            print("Press the button twice to start/stop recording...")
            prompt_printed = True  # Set the flag to True after printing the prompt

        # Double press detection logic
        try:

            # Wait for the first button press
            channel = GPIO.wait_for_edge(BUTTON_PIN, GPIO.RISING, timeout=1000)  # Using a timeout to prevent blocking
            if channel is None:
                continue
            current_time = time.time()
            time_diff = current_time - last_press_time
            if time_diff < DOUBLE_PRESS_MAX_DELAY:
                double_press_detected = True
            else:
                double_press_detected = False
            last_press_time = current_time
            if double_press_detected:
                time.sleep(0.2)
                recording = not recording
                prompt_printed = False

                if recording:
                    print("Recording started...")
                    # Start recording audio
                    recording_data = sd.rec(int(8 * 44100), samplerate=44100, channels=1)
                    # Turn on the LED
                    update_led('ON', Color.RED, 0.75)  # Solid RED while recording
                else:
                    print("Recording stopped. Processing audio...")
                    #update_led('OFF')  # Turn off LED when idle
                    sd.stop()
                    sf.write(audio_path, recording_data, 44100, 'PCM_16')
                    # Turn on Green LED breathing pattern to indicate processing
                    update_led('BREATHE', Color.GREEN, 0.5)  # Green breathing pattern while processing
                    # Transcribe Telugu/Hindi audio to English text using Azure Speech Recognition
                    english_text, detected_audio_language = transcribe_audio_to_text(audio_path)
                    new_message = {"role": "user", "content": english_text}
                    conversation.append(new_message)
                    assistant_reply = generate_response(english_text, conversation, model_name, max_tokens, temperature)
                    new_assistant_message = {"role": "assistant", "content": assistant_reply}
                    conversation.append(new_assistant_message)
                    update_led('OFF')
                    update_led('BLINK', Color.BLUE, 0.1)
                    translate_and_speak(assistant_reply, detected_audio_language, tts_output_path, model_name)
                    update_led('ON', Color.CYAN, 0.1)
                    try:
                        os.remove(audio_path)
                        os.remove(tts_output_path)
                    except Exception as e:
                        print("Error deleting audio files:", str(e))
                        continue
        except RuntimeError as e:
            print("RuntimeError:", str(e))
            continue

except KeyboardInterrupt:
    print("\nScript terminated by user.")

finally:
    # Clean up GPIO
    GPIO.cleanup()
    update_led('OFF')  # Turn off LED when idle
