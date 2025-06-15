import os
import base64
import json
import time
import logging
import sys
import platform # Added for platform detection

# Attempt to import Raspberry Pi specific libraries
IS_RASPBERRY_PI = platform.machine().startswith(('armv', 'aarch64')) and os.path.exists('/proc/device-tree/model') # Corrected os.path.exists

if IS_RASPBERRY_PI:
    try:
        import RPi.GPIO as GPIO
        from aiy.leds import (Leds, Pattern, RgbLeds, Color) # PrivacyLed not used directly here
        print("Successfully imported RPi.GPIO and aiy.leds")
    except ImportError:
        print("Failed to import RPi.GPIO or aiy.leds. GPIO/LED functionality will be disabled.")
        IS_RASPBERRY_PI = False
else:
    print("Not running on a Raspberry Pi. GPIO/LED functionality will be disabled.")
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

from openai import OpenAI
from config import config

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Global LED instance if on RPi
if IS_RASPBERRY_PI:
    leds = None # Will be initialized in main

def update_led(led_state, color=None, brightness=1.0):
    if not IS_RASPBERRY_PI or leds is None:
        return

    if color is None: # Default color if none provided
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

class SimpleVoiceBot:
    def __init__(self):
        """Initialize the voicebot with OpenAI client and configuration."""
        # Initialize OpenAI client - try different configurations
        try:
            # Check for custom base URL and API key
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
                # Fall back to Azure/proxy configuration
                print("Using Azure/Proxy API configuration")
                self.client = OpenAI(
                    api_key=config.azure_api_key,
                    base_url=config.azure_api_base
                )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Audio settings
        self.sample_rate = 44100
        self.channels = 1
        self.audio_format = 'wav'
        self.recording_duration = 10  # max seconds to record - used for non-Pi recording
        
        # Model settings
        self.model = "gpt-4o-mini-audio-preview"  # Use the correct available audio model
        self.voice = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
        
        # Conversation history
        self.conversation_history = []

        if IS_RASPBERRY_PI:
            # GPIO setup
            self.BUTTON_PIN = 23
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            self.DOUBLE_PRESS_MAX_DELAY = 0.5
            self.last_press_time = 0
            self.rpi_recording_active = False
        
        logger.info("SimpleVoiceBot initialized successfully")
    
    def record_audio(self) -> bytes:
        """Record audio from microphone and return as bytes."""
        if IS_RASPBERRY_PI:
            # On RPi, recording is handled by the main loop's button press logic.
            # This method is called after audio is captured to 'user_audio.wav'.
            print("🎤 Raspberry Pi: Processing pre-recorded audio...")
            audio_path = "user_audio.wav" 
            if not os.path.exists(audio_path): # Corrected os.path.exists
                logger.error(f"RPi: {audio_path} not found. Recording might have failed.")
                return b""
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            print("✅ RPi: Recording processed from file")
            return audio_bytes
        else: # Standard (non-RPi) recording
            print("🎤 Recording started... Press Enter to stop recording")
            recording_data = []
            recording_active_flag = True # Use a flag to control callback data append
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                if recording_active_flag:
                    recording_data.append(indata.copy())
            
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback
            )
            
            with stream:
                input("Press Enter to stop recording...")
                recording_active_flag = False # Signal callback to stop appending
            
            if recording_data:
                audio_array = np.concatenate(recording_data, axis=0)
                temp_audio_path = "temp_recording.wav"
                sf.write(temp_audio_path, audio_array, self.sample_rate)
                with open(temp_audio_path, 'rb') as f:
                    audio_bytes = f.read()
                os.remove(temp_audio_path)
                print("✅ Recording completed")
                return audio_bytes
            else:
                logger.warning("No audio data recorded in standard mode.")
                return b""
    
    def encode_audio(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 string."""
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def decode_audio(self, base64_audio: str) -> bytes:
        """Decode base64 audio string to bytes."""
        return base64.b64decode(base64_audio)
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio from bytes."""
        # Save to temporary file and play
        temp_output_path = "temp_response.wav"
        
        try:
            with open(temp_output_path, 'wb') as f:
                f.write(audio_bytes)
            
            # Read and play the audio file
            data, fs = sf.read(temp_output_path)
            print("🔊 Playing response...")
            sd.play(data, fs)
            sd.wait()  # Wait until the audio finishes playing
            print("✅ Audio playback completed")
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
    
    def send_audio_to_gpt(self, audio_bytes: bytes, text_prompt: str = None) -> dict:
        """Send audio to GPT-4o-mini-audio-preview and get response."""
        try:
            # Encode audio to base64
            encoded_audio = self.encode_audio(audio_bytes)
            
            # Prepare message content
            message_content = []
            
            # Add text prompt if provided
            if text_prompt:
                message_content.append({
                    "type": "text",
                    "text": text_prompt
                })
            
            # Add audio input
            message_content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": encoded_audio,
                    "format": self.audio_format
                }
            })
            
            # Prepare messages with conversation history
            messages = self.conversation_history + [{
                "role": "user",
                "content": message_content
            }]
            
            # Make API call
            print("🤖 Processing with GPT-4o-audio-preview...")
            completion = self.client.chat.completions.create(
                model=self.model,
                modalities=["text", "audio"],
                audio={"voice": self.voice, "format": self.audio_format},
                messages=messages
            )
            
            response_message = completion.choices[0].message
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user", 
                "content": text_prompt or "Audio input"
            })
            
            if response_message.audio:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_message.audio.transcript or "Audio response"
                })
            elif response_message.content:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_message.content
                })
            
            print("✅ Response received from GPT-4o")
            return response_message
            
        except Exception as e:
            logger.error(f"Error communicating with GPT: {e}")
            raise
    
    def process_interaction(self, audio_bytes: bytes = None, text_input: str = None):
        """Handles a single interaction: record, send to GPT, play response."""
        try:
            user_provided_audio = bool(audio_bytes)

            if text_input:
                # Handling text input directly (not primary for this voicebot version)
                print(f"👤 You (text): {text_input}")
                # Note: send_audio_to_gpt will handle history for text_prompt
            elif not audio_bytes and not IS_RASPBERRY_PI: # Standard mode, record now
                audio_bytes = self.record_audio()
                if not audio_bytes:
                    print("No audio recorded.")
                    return
            elif not audio_bytes and IS_RASPBERRY_PI:
                # This case should ideally not happen if RPi logic calls with audio_bytes
                print("RPi: process_interaction called without audio_bytes. Waiting for button press.")
                return


            # Send to GPT
            # If audio_bytes were passed (e.g. from RPi), use them.
            # If text_input was provided, it's passed as text_prompt.
            # send_audio_to_gpt handles adding user input to history.
            response_message = self.send_audio_to_gpt(audio_bytes, text_prompt=text_input)

            response_text = None
            response_audio_bytes = None

            if response_message:
                # Extract audio if present
                if hasattr(response_message, 'audio') and response_message.audio and \
                   hasattr(response_message.audio, 'data') and response_message.audio.data:
                    response_audio_bytes = self.decode_audio(response_message.audio.data)

                # Extract text:
                # 1. From audio transcript (priority if audio response exists)
                if hasattr(response_message, 'audio') and response_message.audio and \
                   hasattr(response_message.audio, 'transcript') and response_message.audio.transcript:
                    response_text = response_message.audio.transcript
                
                # 2. From .content if no transcript from .audio or if .content is primary
                if not response_text and hasattr(response_message, 'content'):
                    if isinstance(response_message.content, str):
                        response_text = response_message.content
                    elif isinstance(response_message.content, list): # Handle list of content parts
                        temp_texts = []
                        for item in response_message.content:
                            if hasattr(item, 'type') and item.type == 'text' and hasattr(item, 'text'):
                                # item.text could be an object with .value or a string itself
                                if hasattr(item.text, 'value') and isinstance(item.text.value, str):
                                    temp_texts.append(item.text.value)
                                elif isinstance(item.text, str): 
                                    temp_texts.append(item.text)
                        if temp_texts:
                            response_text = " ".join(temp_texts)
            
            if response_text:
                print(f"🤖 Assistant: {response_text}")
            
            if response_audio_bytes:
                self.play_audio(response_audio_bytes)
            elif response_text and not response_audio_bytes: # Only text, synthesize speech
                print("🔊 Synthesizing text response...")
                try:
                    tts_response = self.client.audio.speech.create(
                        model="tts-1", 
                        voice=self.voice,
                        input=response_text
                    )
                    self.play_audio(tts_response.read())
                except Exception as e:
                    logger.error(f"Error during TTS: {e}")
            elif not response_text and not response_audio_bytes:
                print("🤷 No response content (text or audio) from assistant.")

        except Exception as e:
            logger.error(f"Error during interaction: {e}", exc_info=True)


def main():
    global leds 
    bot = SimpleVoiceBot()

    if IS_RASPBERRY_PI:
        leds = Leds() 
        update_led('ON', Color.CYAN, 0.1) # Idle
        
        rpi_recording_buffer = [] # Buffer for RPi audio data
        rpi_audio_stream = None

        def rpi_mic_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"RPi mic callback status: {status}")
            if bot.rpi_recording_active:
                rpi_recording_buffer.append(indata.copy())

        print("Raspberry Pi VoiceBot activated. Press button twice to start/stop.")
        try:
            while True:
                try:
                    # Non-blocking edge detection with timeout
                    edge_detected = GPIO.wait_for_edge(bot.BUTTON_PIN, GPIO.RISING, timeout=100) 
                    
                    if edge_detected is not None: # Button press detected
                        current_time = time.time()
                        time_diff = current_time - bot.last_press_time
                        
                        if time_diff < bot.DOUBLE_PRESS_MAX_DELAY: # Double press
                            bot.last_press_time = 0 # Reset for next double press detection
                            bot.rpi_recording_active = not bot.rpi_recording_active

                            if bot.rpi_recording_active:
                                print("RPi: Recording started...")
                                rpi_recording_buffer.clear()
                                rpi_audio_stream = sd.InputStream(
                                    samplerate=bot.sample_rate,
                                    channels=bot.channels,
                                    callback=rpi_mic_callback
                                )
                                rpi_audio_stream.start()
                                update_led('ON', Color.RED, 0.75) # Recording
                            else: # Recording stopped
                                print("RPi: Recording stopped. Processing audio...")
                                if rpi_audio_stream:
                                    rpi_audio_stream.stop()
                                    rpi_audio_stream.close()
                                    rpi_audio_stream = None
                                
                                update_led('BREATHE', Color.GREEN, 0.5) # Processing
                                
                                if rpi_recording_buffer:
                                    audio_array = np.concatenate(rpi_recording_buffer, axis=0)
                                    audio_path = "user_audio.wav" # Temp file for RPi audio
                                    sf.write(audio_path, audio_array, bot.sample_rate)
                                    
                                    # Read the recorded audio bytes for process_interaction
                                    with open(audio_path, 'rb') as f_audio:
                                        recorded_audio_bytes = f_audio.read()
                                    
                                    if recorded_audio_bytes:
                                        bot.process_interaction(audio_bytes=recorded_audio_bytes)
                                    else:
                                        print("RPi: No audio data recorded.")
                                    
                                    try: # Clean up the temp audio file
                                        os.remove(audio_path)
                                    except OSError as e:
                                        logger.error(f"Error removing RPi audio file {audio_path}: {e}")
                                else:
                                    print("RPi: No audio data to process.")
                                
                                update_led('ON', Color.CYAN, 0.1) # Back to Idle
                                print("RPi: Press button twice to start/stop.")
                                
                        else: # Single press (or first press of a potential double press)
                            bot.last_press_time = current_time
                    
                    time.sleep(0.01) # Small sleep to yield CPU

                except RuntimeError as e: # Catch GPIO specific errors
                    if "edge detection" in str(e).lower():
                        logger.warning(f"RPi GPIO Error (likely already cleaned up): {e}")
                        # Potentially re-initialize GPIO if necessary, or just log and continue
                        time.sleep(1)
                    else:
                        logger.error(f"RPi Runtime Error: {e}")
                        time.sleep(1)
                    continue 
        
        except KeyboardInterrupt:
            print("\nRPi: Script terminated by user.")
        finally:
            if rpi_audio_stream and rpi_audio_stream.active:
                rpi_audio_stream.stop()
                rpi_audio_stream.close()
            if IS_RASPBERRY_PI: # Ensure GPIO cleanup only if it was setup
                GPIO.cleanup()
                update_led('OFF') # Turn off LEDs
                print("RPi: GPIO cleanup done, LEDs off.")

    else: # Not on Raspberry Pi
        print("Running in standard mode (no GPIO/Button). Press Ctrl+C to exit.")
        try:
            while True:
                print("\\n--- New Interaction (Standard Mode) ---")
                bot.process_interaction()
                # Add a small delay or a specific prompt if needed, e.g.:
                # input("Press Enter to start a new interaction or Ctrl+C to exit...")
                time.sleep(1) # Brief pause before next potential interaction
        except KeyboardInterrupt:
            print("\\nStandard mode: Script terminated by user.")
        finally:
            print("Standard mode interaction loop finished.")

if __name__ == "__main__":
    main()
