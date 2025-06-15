import os
import base64
import json
import time
import logging
import sys
import subprocess 
from openai import OpenAI
from config import config
import platform

# Raspberry Pi specific libraries
try:
    import RPi.GPIO as GPIO
    from aiy.leds import (Leds, Pattern, RgbLeds, Color)
    print("Successfully imported RPi.GPIO and aiy.leds")
    IS_RASPBERRY_PI = True
except ImportError:
    print("Failed to import RPi.GPIO or aiy.leds. GPIO/LED functionality will be disabled.")
    IS_RASPBERRY_PI = False # Should ideally not happen if this script is RPi-only

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Global LED instance
if IS_RASPBERRY_PI:
    leds = None # Will be initialized in main

def update_led(led_state, color=None, brightness=1.0):
    if not IS_RASPBERRY_PI or leds is None:
        return

    if color is None: 
        color = Color.WHITE 

    brightness = max(0.0, min(brightness, 1.0))
    scaled_color = tuple(int(c * brightness) for c in color)

    if led_state == 'ON':
        leds.update(Leds.rgb_on(scaled_color))
    elif led_state == 'OFF':
        leds.update(Leds.rgb_off())
    elif led_state == 'BLINK':
        leds.pattern = Pattern.blink(500) 
        leds.update(Leds.rgb_pattern(scaled_color))
    elif led_state == 'BREATHE':
        leds.pattern = Pattern.breathe(1500)
        leds.update(Leds.rgb_pattern(scaled_color))

class SimpleVoiceBotRPi:
    def __init__(self):
        """Initialize the RPi voicebot with OpenAI client and configuration."""
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            openai_base_url = os.getenv('OPENAI_BASE_URL')
            
            if openai_api_key and openai_base_url:
                print(f"Using custom OpenAI API endpoint: {openai_base_url}")
                self.client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_base_url
                )
            elif openai_api_key:
                print("Using direct OpenAI API")
                self.client = OpenAI(api_key=openai_api_key)
            else:
                print("Using Azure/Proxy API configuration")
                self.client = OpenAI(
                    api_key=config.azure_api_key,
                    base_url=config.azure_api_base
                )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        self.audio_format = 'wav' # arecord/aplay typically use wav
        self.rpi_audio_file = "user_audio.wav" # File for arecord
        
        self.model = "gpt-4o-mini-audio-preview"
        self.voice = "alloy"
        
        self.conversation_history = []

        if IS_RASPBERRY_PI:
            self.BUTTON_PIN = 23
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            self.DOUBLE_PRESS_MAX_DELAY = 0.5
            self.last_press_time = 0
            self.rpi_recording_active = False # To track arecord process
        
        logger.info("SimpleVoiceBotRPi initialized successfully")
    
    def record_audio_from_file(self) -> bytes:
        """Reads audio from the file saved by arecord."""
        print(f"🎤 Processing audio from {self.rpi_audio_file}...")
        if not os.path.exists(self.rpi_audio_file):
            logger.error(f"{self.rpi_audio_file} not found. Recording might have failed.")
            return b""
        with open(self.rpi_audio_file, 'rb') as f:
            audio_bytes = f.read()
        print(f"✅ Audio processed from {self.rpi_audio_file}")
        return audio_bytes
    
    def encode_audio(self, audio_bytes: bytes) -> str:
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def decode_audio(self, base64_audio: str) -> bytes:
        return base64.b64decode(base64_audio)
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio from bytes using aplay."""
        temp_output_path = "temp_response.wav"
        
        try:
            with open(temp_output_path, 'wb') as f:
                f.write(audio_bytes)
            
            print("🔊 Playing response using aplay...")
            # Use '-D hw:CARD=seeed2micvoicec,DEV=0' or your specific card if default doesn't work
            # Find your card with 'aplay -l'
            aplay_command = ['aplay', temp_output_path] 
            playback_process = subprocess.run(aplay_command, capture_output=True, text=True)
            if playback_process.returncode != 0:
                logger.error(f"aplay error: {playback_process.stderr}")
            else:
                print("✅ Audio playback completed via aplay")
            
        except Exception as e:
            logger.error(f"Error playing audio with aplay: {e}")
        finally:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
    
    def send_audio_to_gpt(self, audio_bytes: bytes, text_prompt: str = None) -> dict:
        try:
            encoded_audio = self.encode_audio(audio_bytes)
            message_content = []
            if text_prompt:
                message_content.append({"type": "text", "text": text_prompt})
            
            message_content.append({
                "type": "image_url", # This seems incorrect for audio, should be audio related
                                     # However, current OpenAI API for gpt-4o with audio input
                                     # might still use a similar structure or a specific SDK call.
                                     # For direct audio chat, the API might differ.
                                     # This part needs to align with the exact API spec for audio input.
                                     # Assuming a placeholder or a simplified approach for now.
                                     # The correct method is client.audio.transcriptions.create for speech-to-text
                                     # and client.audio.speech.create for text-to-speech.
                                     # This function seems to intend to send audio *for chat completion directly*.
                "image_url": { # Placeholder - this is for images, not audio.
                    "url": f"data:audio/{self.audio_format};base64,{encoded_audio}"
                }
            })

            # Let's correct the GPT interaction for audio
            # 1. Transcribe audio to text
            # 2. Send text to chat completion
            # 3. Get text response
            # 4. Convert text response to speech
            
            # For simplicity in this direct modification, we'll assume the user wants to send audio
            # and get a text response, then convert that text response to audio.
            # The original script's send_audio_to_gpt was structured for multimodal input,
            # which might not be the primary goal here or might be misusing the API structure.

            # Corrected approach:
            # 1. Transcribe audio to text
            temp_audio_for_transcription = "transcribe_temp.wav"
            with open(temp_audio_for_transcription, 'wb') as f:
                f.write(audio_bytes)

            with open(temp_audio_for_transcription, 'rb') as audio_file_obj:
                transcription_response = self.client.audio.transcriptions.create(
                    model="whisper-1", # Or other transcription model
                    file=audio_file_obj
                )
            os.remove(temp_audio_for_transcription)
            user_text = transcription_response.text
            print(f"🎤 You said: {user_text}")

            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_text})
            if text_prompt: # If there was an initial system-like prompt for the audio
                 self.conversation_history.insert(-1, {"role": "system", "content": text_prompt})


            # 2. Send text to chat model
            update_led('BREATHE', Color.BLUE) # Thinking
            chat_response = self.client.chat.completions.create(
                model=self.model.replace("-audio-preview", ""), # Use text model
                messages=self.conversation_history
            )
            assistant_response_text = chat_response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_response_text})
            print(f"🤖 Assistant: {assistant_response_text}")
            update_led('OFF')

            # 3. Convert text response to speech
            update_led('ON', Color.YELLOW) # Speaking
            speech_response = self.client.audio.speech.create(
                model="tts-1", # Or other TTS model
                voice=self.voice,
                input=assistant_response_text
            )
            assistant_audio_bytes = speech_response.read()
            update_led('OFF')
            
            return {"text_response": assistant_response_text, "audio_response": assistant_audio_bytes}

        except Exception as e:
            logger.error(f"Error in GPT interaction: {e}")
            update_led('OFF')
            return {"text_response": "Sorry, I encountered an error.", "audio_response": None}

def main():
    global leds # Allow main to initialize leds
    if IS_RASPBERRY_PI:
        try:
            leds = RgbLeds() # Initialize LEDs
            update_led('ON', Color.GREEN, 0.3) # Initial ready state
            print("LEDs initialized.")
        except Exception as e:
            print(f"Failed to initialize LEDs: {e}")
            # Continue without LED support if initialization fails

    bot = SimpleVoiceBotRPi()
    print("RPi Voice Bot is ready. Press the button to start recording.")
    print("A single press starts/stops recording. A double press exits.")

    last_button_state = GPIO.LOW
    recording_process = None # To store the subprocess for arecord

    try:
        while True:
            if not IS_RASPBERRY_PI:
                print("This script is intended for Raspberry Pi with a button. Exiting.")
                break

            button_state = GPIO.input(bot.BUTTON_PIN)
            current_time = time.time()

            if button_state == GPIO.HIGH and last_button_state == GPIO.LOW: # Button pressed
                time.sleep(0.05) # Debounce

                if bot.rpi_recording_active: # If recording, stop it
                    print("Button pressed. Stopping recording...")
                    update_led('OFF')
                    if recording_process:
                        recording_process.terminate() # Send SIGTERM to arecord
                        try:
                            recording_process.wait(timeout=5) # Wait for it to finish
                        except subprocess.TimeoutExpired:
                            recording_process.kill() # Force kill if it doesn't stop
                        print("arecord process stopped.")
                    bot.rpi_recording_active = False
                    
                    # Process the recorded audio
                    audio_data = bot.record_audio_from_file()
                    if audio_data:
                        update_led('BREATHE', Color.CYAN, 0.7) # Processing
                        # For RPi, the initial prompt is fixed or could be configured elsewhere
                        initial_prompt = "You are a helpful assistant on a Raspberry Pi."
                        response = bot.send_audio_to_gpt(audio_data, text_prompt=initial_prompt)
                        update_led('OFF')
                        
                        if response and response.get("audio_response"):
                            update_led('ON', Color.MAGENTA, 0.7) # Speaking
                            bot.play_audio(response["audio_response"])
                            update_led('ON', Color.GREEN, 0.3) # Ready again
                        else:
                            print("No audio response to play.")
                            update_led('ON', Color.RED, 0.5) # Error indication
                            time.sleep(2)
                            update_led('ON', Color.GREEN, 0.3) # Ready again
                    else:
                        print("No audio data captured to process.")
                        update_led('ON', Color.RED, 0.5) 
                        time.sleep(1)
                        update_led('ON', Color.GREEN, 0.3)

                else: # If not recording, start it
                    print("Button pressed. Starting recording...")
                    update_led('ON', Color.RED) # Recording indicator
                    # Use '-D hw:CARD=seeed2micvoicec,DEV=0' or your specific card if default doesn't work
                    # Find your card with 'arecord -l'
                    # Using a common format, adjust if needed: -f S16_LE -r 16000
                    arecord_command = [
                        'arecord', 
                        '-D', 'plughw:1,0', # Example, replace with your device from 'arecord -L'
                        '-f', 'S16_LE',    # Sample format (16-bit little-endian)
                        '-r', '16000',     # Sample rate (16kHz)
                        '-c', '1',         # Channels (mono)
                        '--duration=10', # Max duration, but button press will stop it sooner
                        bot.rpi_audio_file
                    ]
                    try:
                        # Start arecord in the background
                        recording_process = subprocess.Popen(arecord_command)
                        bot.rpi_recording_active = True
                        print(f"arecord started with PID: {recording_process.pid}. Recording to {bot.rpi_audio_file}")
                        print("Press button again to stop.")
                    except FileNotFoundError:
                        logger.error("arecord command not found. Please ensure it's installed and in PATH.")
                        update_led('BLINK', Color.RED) # Error
                        time.sleep(3)
                        update_led('ON', Color.GREEN, 0.3) # Ready
                    except Exception as e:
                        logger.error(f"Failed to start arecord: {e}")
                        update_led('BLINK', Color.RED)
                        time.sleep(3)
                        update_led('ON', Color.GREEN, 0.3)


                # Double press detection for exit
                if current_time - bot.last_press_time < bot.DOUBLE_PRESS_MAX_DELAY:
                    print("Double press detected. Exiting.")
                    update_led('BLINK', Color.YELLOW, 0.5)
                    time.sleep(1)
                    update_led('OFF')
                    if bot.rpi_recording_active and recording_process:
                        recording_process.terminate()
                        try:
                            recording_process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            recording_process.kill()
                    break 
                bot.last_press_time = current_time

            last_button_state = button_state
            time.sleep(0.01) # Small delay to prevent busy-waiting

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if IS_RASPBERRY_PI:
            if bot.rpi_recording_active and recording_process: # Ensure arecord is stopped
                print("Cleaning up arecord process...")
                recording_process.terminate()
                try:
                    recording_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    recording_process.kill()
            if os.path.exists(bot.rpi_audio_file): # Clean up audio file
                 os.remove(bot.rpi_audio_file)
            update_led('OFF') # Turn off LED
            GPIO.cleanup() # Clean up GPIO resources
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
