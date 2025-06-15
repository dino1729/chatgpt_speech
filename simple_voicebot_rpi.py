# Load environment variables from .env file (ensure this is at the very top)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv is not installed. Please run: pip install python-dotenv")
    exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("openai is not installed. Please run: pip install openai")
    exit(1)

import os
import base64
import json
import time
import logging
import sys
import subprocess 
import platform

# Raspberry Pi specific libraries
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Failed to import RPi.GPIO. GPIO functionality will be disabled.")
    GPIO_AVAILABLE = False

try:
    from aiy.leds import (Leds, Pattern, RgbLeds, Color)
    LEDS_AVAILABLE = True
except ImportError:
    print("Failed to import aiy.leds. LED functionality will be disabled.")
    LEDS_AVAILABLE = False
    # Define dummy Color class for compatibility
    class Color:
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        YELLOW = (255, 255, 0)
        CYAN = (0, 255, 255)
        MAGENTA = (255, 0, 255)

# Check if we're actually on a Raspberry Pi
IS_RASPBERRY_PI = platform.system() == 'Linux' and ('arm' in platform.machine().lower() or 'aarch64' in platform.machine().lower())

if IS_RASPBERRY_PI:
    print("Running on Raspberry Pi")
    if GPIO_AVAILABLE:
        print("GPIO functionality enabled")
    if LEDS_AVAILABLE:
        print("LED functionality enabled")
else:
    print("Not running on Raspberry Pi - some features may be limited")

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Global LED instance
leds = None # Will be initialized in main if available

def update_led(led_state, color=None, brightness=1.0):
    if not LEDS_AVAILABLE or leds is None:
        # Print LED state for debugging when LEDs are not available
        print(f"LED: {led_state} {color if color else 'DEFAULT'}")
        return

    if color is None: 
        color = Color.WHITE 

    brightness = max(0.0, min(brightness, 1.0))
    scaled_color = tuple(int(c * brightness) for c in color)

    try:
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
    except Exception as e:
        print(f"LED error: {e}")

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
                raise ValueError(
                    "No OpenAI API configuration found. Please set OPENAI_API_KEY environment variable "
                    "or create a config.py file with azure_api_key and azure_api_base."
                )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            print("\nAPI Configuration Help:")
            print("1. Set environment variable: export OPENAI_API_KEY='your-key-here'")
            print("2. Or create config.py with azure_api_key and azure_api_base")
            print("3. Make sure your API key is valid and has sufficient credits")
            raise
        
        self.audio_format = 'wav' # arecord/aplay typically use wav
        self.rpi_audio_file = "user_audio.wav" # File for arecord
        self.audio_device = None  # Will be set during initialization
        
        self.model = "gpt-4o-mini-audio-preview"
        self.voice = "alloy"
        
        self.conversation_history = []

        # GPIO setup only if available
        if IS_RASPBERRY_PI and GPIO_AVAILABLE:
            self.BUTTON_PIN = 23
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            self.DOUBLE_PRESS_MAX_DELAY = 0.5
            self.last_press_time = 0
            self.rpi_recording_active = False # To track arecord process
            print("GPIO button initialized on pin 23")
        else:
            print("GPIO not available - will use keyboard input instead")
            self.rpi_recording_active = False
        
        # Set up audio device
        if IS_RASPBERRY_PI:
            self.audio_device = self.get_audio_device()
        else:
            self.audio_device = 'default'
        
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

    def get_audio_device(self):
        """Get the best available audio recording device."""
        # Common audio device options to try in order of preference
        devices_to_try = [
            'default',           # System default
            'plughw:1,0',       # USB device
            'plughw:0,0',       # Built-in device  
            'hw:1,0',           # Direct hardware access
            'hw:0,0',           # Built-in direct access
        ]
        
        for device in devices_to_try:
            test_command = [
                'arecord', 
                '-D', device,
                '-f', 'S16_LE',
                '-r', '16000',
                '-c', '1',
                '--duration=1',
                '/tmp/test_device.wav'
            ]
            
            try:
                result = subprocess.run(test_command, capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    if os.path.exists('/tmp/test_device.wav'):
                        os.remove('/tmp/test_device.wav')
                    print(f"Using audio device: {device}")
                    return device
            except:
                continue
        
        print("No working audio device found, using 'default'")
        return 'default'

def detect_audio_devices():
    """Detect available audio recording devices."""
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Available audio recording devices:")
            print(result.stdout)
            return True
        else:
            print("No audio recording devices found or arecord not available")
            return False
    except FileNotFoundError:
        print("arecord command not found. Please install alsa-utils:")
        print("sudo apt update && sudo apt install alsa-utils")
        return False
    except Exception as e:
        print(f"Error detecting audio devices: {e}")
        return False

def check_audio_setup():
    """Check if audio recording is properly set up."""
    print("Checking audio setup...")
    
    # Check if arecord is available
    if not detect_audio_devices():
        return False
    
    # Try to record a short test
    test_file = "test_audio.wav"
    try:
        print("Testing audio recording (2 seconds)...")
        test_command = [
            'arecord', 
            '-D', 'default',  # Use default device for initial test
            '-f', 'S16_LE',
            '-r', '16000',
            '-c', '1',
            '--duration=2',
            test_file
        ]
        
        result = subprocess.run(test_command, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 and os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            os.remove(test_file)  # Clean up
            if file_size > 1000:  # Should be more than 1KB for 2 seconds
                print("✓ Audio recording test successful!")
                return True
            else:
                print("✗ Audio recording test failed - file too small")
                return False
        else:
            print(f"✗ Audio recording test failed: {result.stderr}")
            if "Device or resource busy" in result.stderr:
                print("Try a different audio device. Run 'arecord -l' to see available devices.")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Audio recording test timed out")
        return False
    except Exception as e:
        print(f"✗ Audio recording test error: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def check_requirements():
    """Check if all required dependencies are available."""
    print("Checking requirements...")
    
    missing_requirements = []
    
    # Check Python packages
    try:
        import openai
        print("✓ openai package available")
    except ImportError:
        missing_requirements.append("openai")
    
    # Check system commands
    try:
        subprocess.run(['arecord', '--version'], capture_output=True, check=True)
        print("✓ arecord available")
    except (FileNotFoundError, subprocess.CalledProcessError):
        missing_requirements.append("alsa-utils (for arecord)")
    
    try:
        subprocess.run(['aplay', '--version'], capture_output=True, check=True)
        print("✓ aplay available")
    except (FileNotFoundError, subprocess.CalledProcessError):
        missing_requirements.append("alsa-utils (for aplay)")
    
    if missing_requirements:
        print("\n❌ Missing requirements:")
        for req in missing_requirements:
            print(f"  - {req}")
        print("\nTo install missing packages:")
        if "openai" in missing_requirements:
            print("  pip install openai")
        if "alsa-utils" in [req for req in missing_requirements if "alsa-utils" in req]:
            print("  sudo apt update && sudo apt install alsa-utils")
        return False
    
    print("✓ All requirements satisfied")
    return True

def main():
    global leds # Allow main to initialize leds
    
    print("=== RPi Voice Bot Startup ===")
    
    # Check requirements first
    if not check_requirements():
        print("\n⚠️  Some requirements are missing. The bot may not work properly.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            return
    
    # Initialize LEDs if available
    if IS_RASPBERRY_PI and LEDS_AVAILABLE:
        try:
            leds = RgbLeds(18) # GPIO 18 for RGB LED
            print("LEDS initialized")
        except Exception as e:
            print(f"Failed to initialize LEDS: {e}")
            LEDS_AVAILABLE = False
    
    # Main loop
    while True:
        # Record audio
        audio_bytes = b""
