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
import io

# Import LLM audio functions from simple_voicebot.py
# This provides: encode_audio, decode_audio, send_audio_to_gpt methods
from simple_voicebot import SimpleVoiceBot

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

class SimpleVoiceBotRPi(SimpleVoiceBot):
    def __init__(self):
        """Initialize the RPi voicebot with OpenAI client and configuration."""
        # Initialize parent class
        super().__init__()
        
        # Override audio settings for RPi
        self.audio_format = 'wav' # arecord/aplay typically use wav
        self.rpi_audio_file = "user_audio.wav" # File for arecord
        self.audio_device = None  # Will be set during initialization
        
        # GPIO setup only if available
        if IS_RASPBERRY_PI and GPIO_AVAILABLE:
            self.BUTTON_PIN = 23
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            self.DEBOUNCE_DELAY = 0.3  # Debounce delay in seconds
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
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio from bytes using aplay (RPi-specific override)."""
        temp_output_path = "temp_response.wav"
        update_led('BLINK', Color.BLUE, 0.75)  # Bot responding
        
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
    
    def process_interaction(self, audio_bytes: bytes = None):
        """Handles a single interaction: send audio to GPT and play response."""
        try:
            if not audio_bytes:
                print("No audio data provided.")
                return

            # Send to GPT using inherited method
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
        
        print("Raspberry Pi VoiceBot activated. Press button to start/stop recording.")
        
        try:
            while True:
                try:
                    # Non-blocking edge detection with timeout
                    edge_detected = GPIO.wait_for_edge(bot.BUTTON_PIN, GPIO.RISING, timeout=100)
                    
                    if edge_detected is not None:  # Button press detected
                        current_time = time.time()
                        
                        # Debounce logic
                        if (current_time - bot.last_press_time) > bot.DEBOUNCE_DELAY:
                            bot.last_press_time = current_time  # Update last_press_time for debounce

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
                                print("Press button to start/stop recording.")
                                
                # Removed the old 'else' block for single press of a double press
                    
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
            print("Executing cleanup actions...")
            # Clean up arecord process
            if hasattr(bot, 'arecord_process') and bot.arecord_process and bot.arecord_process.poll() is None:
                print("Terminating active arecord process...")
                bot.arecord_process.terminate()
                try:
                    print("Waiting for arecord process to terminate (timeout 0.5s)...")
                    bot.arecord_process.wait(timeout=0.5) 
                    print("arecord process terminated.")
                except subprocess.TimeoutExpired:
                    print("arecord process timed out, attempting to kill.")
                    if bot.arecord_process.poll() is None: # Check if still running before kill
                        bot.arecord_process.kill()
                        try:
                            bot.arecord_process.wait(timeout=0.5) # Wait briefly for kill
                            print("arecord process killed.")
                        except: # Ignore errors on wait after kill
                            print("arecord process killed (wait failed or already dead).")
                except KeyboardInterrupt:
                    print("KeyboardInterrupt during arecord_process.wait() in finally. Attempting to kill.")
                    if bot.arecord_process.poll() is None: # Check if still running before kill
                        bot.arecord_process.kill()
                    # Continue to other cleanup steps
                except Exception as e:
                    print(f"Error during arecord_process cleanup in finally: {e}")
            
            # GPIO cleanup
            if GPIO_AVAILABLE:
                try:
                    print("Cleaning up GPIO...")
                    GPIO.cleanup()
                    print("GPIO cleanup successful.")
                except Exception as e:
                    print(f"Warning: Error during GPIO.cleanup(): {e}. This might be okay if already cleaned or not setup.")
            
            # LED cleanup
            if LEDS_AVAILABLE and leds is not None: # Ensure leds object was initialized
                try:
                    print("Turning LEDs off...")
                    update_led('OFF')
                    print("LEDs should be off.")
                except Exception as e:
                    print(f"Error turning off LEDs: {e}")
            elif not LEDS_AVAILABLE:
                print("LEDs not available, skipping LED off command.")
            elif leds is None:
                print("LEDs object not initialized, skipping LED off command.")
            
            print("Cleanup process completed.")

    else:  # Not on Raspberry Pi or GPIO not available
        print("❌ GPIO not available. This script is designed for Raspberry Pi hardware with button input.")
        print("Exiting... This is a headless voicebot that requires hardware button interaction.")
        update_led('OFF')
        return


if __name__ == "__main__":
    main()
