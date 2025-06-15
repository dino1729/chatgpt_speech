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
                print("⚠️  Note: Custom endpoints may not support GPT-4o audio features.")
                print("   The bot will automatically fall back to transcription + TTS if needed.")
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
            self.arecord_process = None  # Track arecord subprocess
            print("GPIO button initialized on pin 23")
        else:
            print("GPIO not available - headless mode requires hardware button")
            self.rpi_recording_active = False
            self.arecord_process = None
        
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
    
    def send_audio_to_gpt(self, audio_bytes: bytes) -> dict:
        """Send audio to GPT-4o-audio-preview and get response."""
        try:
            # Check if we can use the new audio API
            try:
                # Try the new audio API first
                return self._send_audio_new_api(audio_bytes)
            except (TypeError, Exception) as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["modalities", "input_audio", "unexpected keyword", "audio"]):
                    print("⚠️  Audio API not supported by this endpoint, falling back to transcription + TTS...")
                    return self._send_audio_fallback(audio_bytes)
                else:
                    raise e
        except Exception as e:
            logger.error(f"Error communicating with GPT: {e}")
            update_led('OFF')
            raise

    def _send_audio_new_api(self, audio_bytes: bytes) -> dict:
        """Send audio using the new GPT-4o-audio API."""
        # Encode audio to base64
        encoded_audio = self.encode_audio(audio_bytes)
        
        # Prepare message content - only audio input
        message_content = [{
            "type": "input_audio",
            "input_audio": {
                "data": encoded_audio,
                "format": self.audio_format
            }
        }]
        
        # Prepare messages with conversation history
        messages = self.conversation_history + [{
            "role": "user",
            "content": message_content
        }]
        
        # Make API call
        print("🤖 Processing with GPT-4o-audio-preview...")
        update_led('BREATHE', Color.BLUE, 0.5)  # Processing
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
            "content": "Audio input"
        })
        
        # Handle the response based on the new format
        if hasattr(response_message, 'audio') and response_message.audio:
            # Audio response with transcript
            self.conversation_history.append({
                "role": "assistant",
                "content": response_message.audio.transcript or "Audio response"
            })
        elif response_message.content:
            # Text-only response
            self.conversation_history.append({
                "role": "assistant",
                "content": response_message.content
            })
        
        print("✅ Response received from GPT-4o")
        update_led('OFF')
        return response_message

    def _send_audio_fallback(self, audio_bytes: bytes) -> dict:
        """Fallback method using transcription + text chat + TTS."""
        # Step 1: Transcribe audio to text
        print("🎤 Transcribing audio...")
        update_led('BREATHE', Color.YELLOW, 0.5)  # Transcribing
        
        # Save audio to temporary file for transcription
        temp_audio_path = "temp_transcribe.wav"
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_bytes)
        
        try:
            # Use OpenAI Whisper for transcription
            with open(temp_audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            user_text = transcript.text
            print(f"👤 User said: {user_text}")
            
            # Step 2: Send text to GPT
            print("🤖 Processing with GPT...")
            update_led('BREATHE', Color.BLUE, 0.5)  # Processing
            
            messages = self.conversation_history + [{
                "role": "user",
                "content": user_text
            }]
            
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use standard text model for fallback
                messages=messages
            )
            
            response_text = completion.choices[0].message.content
            print(f"🤖 Assistant: {response_text}")
            
            # Step 3: Convert response to speech
            print("🔊 Converting response to speech...")
            update_led('BREATHE', Color.GREEN, 0.5)  # TTS
            
            speech_response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=response_text,
                response_format="wav"  # Match the audio format
            )
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_text
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Create a mock response object similar to the new API
            class MockResponse:
                def __init__(self, text, audio_data):
                    self.content = text
                    self.audio = MockAudio(audio_data, text)
            
            class MockAudio:
                def __init__(self, audio_data, transcript):
                    self.data = self.encode_audio(audio_data)
                    self.transcript = transcript
                
                def encode_audio(self, audio_bytes):
                    return base64.b64encode(audio_bytes).decode('utf-8')
            
            print("✅ Response generated via fallback method")
            update_led('OFF')
            return MockResponse(response_text, speech_response.content)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def process_interaction(self, audio_bytes: bytes = None):
        """Handles a single interaction: send audio to GPT and play response."""
        try:
            if not audio_bytes:
                print("No audio data provided.")
                return

            # Send to GPT
            response_message = self.send_audio_to_gpt(audio_bytes)

            response_text = None
            response_audio_bytes = None

            if response_message:
                # Extract audio if present (new format)
                if hasattr(response_message, 'audio') and response_message.audio and \
                   hasattr(response_message.audio, 'data') and response_message.audio.data:
                    response_audio_bytes = self.decode_audio(response_message.audio.data)

                # Extract text from transcript or content
                if hasattr(response_message, 'audio') and response_message.audio and \
                   hasattr(response_message.audio, 'transcript') and response_message.audio.transcript:
                    response_text = response_message.audio.transcript
                elif hasattr(response_message, 'content') and response_message.content:
                    if isinstance(response_message.content, str):
                        response_text = response_message.content
                    elif isinstance(response_message.content, list):
                        # Handle list of content parts
                        temp_texts = []
                        for item in response_message.content:
                            if hasattr(item, 'type') and item.type == 'text' and hasattr(item, 'text'):
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
            elif not response_audio_bytes:
                print("🤷 No audio response from assistant.")

        except Exception as e:
            logger.error(f"Error during interaction: {e}", exc_info=True)

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
    global leds
    
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
            leds = Leds()
            print("LEDs initialized")
        except Exception as e:
            print(f"Failed to initialize LEDs: {e}")
            leds = None
    
    # Initialize the bot
    bot = SimpleVoiceBotRPi()
    
    if IS_RASPBERRY_PI and GPIO_AVAILABLE:
        update_led('ON', Color.CYAN, 0.1)  # Idle
        
        print("Raspberry Pi VoiceBot activated. Press button twice to start/stop recording.")
        
        try:
            while True:
                try:
                    # Non-blocking edge detection with timeout
                    edge_detected = GPIO.wait_for_edge(bot.BUTTON_PIN, GPIO.RISING, timeout=100)
                    
                    if edge_detected is not None:  # Button press detected
                        current_time = time.time()
                        time_diff = current_time - bot.last_press_time
                        
                        if time_diff < bot.DOUBLE_PRESS_MAX_DELAY:  # Double press
                            bot.last_press_time = 0  # Reset for next double press detection
                            bot.rpi_recording_active = not bot.rpi_recording_active

                            if bot.rpi_recording_active:
                                print("🎤 Recording started...")
                                update_led('ON', Color.RED, 0.75)  # Recording
                                
                                # Start arecord process
                                arecord_command = [
                                    'arecord',
                                    '-D', bot.audio_device,
                                    '-f', 'S16_LE',
                                    '-r', '44100',  # Match sample rate
                                    '-c', '1',
                                    bot.rpi_audio_file
                                ]
                                
                                bot.arecord_process = subprocess.Popen(
                                    arecord_command, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE
                                )
                                
                            else:  # Recording stopped
                                print("🎤 Recording stopped. Processing audio...")
                                update_led('BREATHE', Color.GREEN, 0.5)  # Processing
                                
                                if hasattr(bot, 'arecord_process') and bot.arecord_process:
                                    bot.arecord_process.terminate()
                                    bot.arecord_process.wait()
                                    bot.arecord_process = None
                                
                                # Process the recorded audio
                                audio_bytes = bot.record_audio_from_file()
                                
                                if audio_bytes:
                                    bot.process_interaction(audio_bytes=audio_bytes)
                                else:
                                    print("No audio data recorded.")
                                
                                # Clean up the temp audio file
                                try:
                                    if os.path.exists(bot.rpi_audio_file):
                                        os.remove(bot.rpi_audio_file)
                                except OSError as e:
                                    logger.error(f"Error removing audio file {bot.rpi_audio_file}: {e}")
                                
                                update_led('ON', Color.CYAN, 0.1)  # Back to idle
                                print("Press button twice to start/stop recording.")
                                
                        else:  # Single press (or first press of a potential double press)
                            bot.last_press_time = current_time
                    
                    time.sleep(0.01)  # Small sleep to yield CPU

                except RuntimeError as e:  # Catch GPIO specific errors
                    if "edge detection" in str(e).lower():
                        logger.warning(f"GPIO Error (likely already cleaned up): {e}")
                        time.sleep(1)
                    else:
                        logger.error(f"Runtime Error: {e}")
                        time.sleep(1)
                    continue
        
        except KeyboardInterrupt:
            print("\nScript terminated by user.")
        finally:
            # Clean up
            if hasattr(bot, 'arecord_process') and bot.arecord_process:
                bot.arecord_process.terminate()
                bot.arecord_process.wait()
            if GPIO_AVAILABLE:
                GPIO.cleanup()
            update_led('OFF')
            print("GPIO cleanup done, LEDs off.")

    else:  # Not on Raspberry Pi or GPIO not available
        print("❌ GPIO not available. This script is designed for Raspberry Pi hardware with button input.")
        print("Exiting... This is a headless voicebot that requires hardware button interaction.")
        update_led('OFF')
        return


if __name__ == "__main__":
    main()
