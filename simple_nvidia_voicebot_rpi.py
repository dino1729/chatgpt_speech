# Load environment variables from .env file (ensure this is at the very top)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv is not installed. Please run: pip install python-dotenv")
    exit(1)

import os
import time
import logging
import sys
import subprocess 
import platform
import io
import yaml
import numpy as np
import soundfile as sf

# Import NVIDIA Riva client for ASR and TTS
try:
    import riva.client as riva
    RIVA_AVAILABLE = True
except ImportError:
    print("nvidia-riva-client is not installed. Please run: pip install nvidia-riva-client")
    RIVA_AVAILABLE = False

# Import OpenAI for LiteLLM text generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("openai is not installed. Please run: pip install openai")
    OPENAI_AVAILABLE = False

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
        """Initialize the RPi voicebot with NVIDIA models and LiteLLM."""
        
        # Load system prompt from config
        try:
            with open('config/prompts.yml', 'r') as f:
                prompts_config = yaml.safe_load(f)
                self.system_prompt = prompts_config.get('system_prompt_content', 
                    "You are a world-class knowledgeable AI voice assistant, Orion, hosted on a Raspberry Pi Zero W2. "
                    "Your mission is to assist users with any questions or tasks they have on a wide range of topics. "
                    "Use your knowledge, skills, and resources to provide accurate, relevant, and helpful responses. "
                    "Please remember that you are a voice assistant and keep answers brief, concise and within 1-2 sentences, "
                    "unless it's absolutely necessary to give a longer response. Be polite, friendly, and respectful in your "
                    "interactions, and try to satisfy the user's needs as best as you can.")
        except Exception as e:
            logger.warning(f"Could not load system prompt from config: {e}")
            self.system_prompt = (
                "You are a world-class knowledgeable AI voice assistant, Orion, hosted on a Raspberry Pi Zero W2. "
                "Your mission is to assist users with any questions or tasks they have on a wide range of topics. "
                "Use your knowledge, skills, and resources to provide accurate, relevant, and helpful responses. "
                "Please remember that you are a voice assistant and keep answers brief, concise and within 1-2 sentences, "
                "unless it's absolutely necessary to give a longer response. Be polite, friendly, and respectful in your "
                "interactions, and try to satisfy the user's needs as best as you can."
            )
        
        # Get API keys from environment
        self.nvidia_api_key = os.getenv('NVIDIA_NIM_API_KEY')
        self.litellm_api_key = os.getenv('LITELLM_API_KEY')
        self.litellm_base_url = os.getenv('LITELLM_BASE_URL')
        
        if not self.nvidia_api_key:
            logger.error("NVIDIA_NIM_API_KEY not found in environment variables")
            raise ValueError("NVIDIA_NIM_API_KEY is required")
        
        if not self.litellm_api_key or not self.litellm_base_url:
            logger.error("LITELLM_API_KEY or LITELLM_BASE_URL not found in environment variables")
            raise ValueError("LITELLM_API_KEY and LITELLM_BASE_URL are required")
        
        # Initialize NVIDIA Riva clients
        if RIVA_AVAILABLE:
            # ASR (Speech to Text) - Parakeet CTC 1.1B
            self.asr_auth = riva.Auth(
                uri='grpc.nvcf.nvidia.com:443',
                use_ssl=True,
                metadata_args=[
                    ['function-id', '1598d209-5e27-4d3c-8079-4751568b1081'],
                    ['authorization', f'Bearer {self.nvidia_api_key}']
                ]
            )
            self.asr_service = riva.ASRService(self.asr_auth)
            
            # TTS (Text to Speech) - Magpie TTS Multilingual
            self.tts_auth = riva.Auth(
                uri='grpc.nvcf.nvidia.com:443',
                use_ssl=True,
                metadata_args=[
                    ['function-id', '877104f7-e885-42b9-8de8-f6e4c6303969'],
                    ['authorization', f'Bearer {self.nvidia_api_key}']
                ]
            )
            self.tts_service = riva.SpeechSynthesisService(self.tts_auth)
            logger.info("NVIDIA Riva services initialized")
        else:
            logger.error("NVIDIA Riva client not available")
            raise ImportError("nvidia-riva-client is required")
        
        # Initialize LiteLLM client
        if OPENAI_AVAILABLE:
            self.llm_client = OpenAI(
                api_key=self.litellm_api_key,
                base_url=self.litellm_base_url
            )
            logger.info("LiteLLM client initialized")
        else:
            logger.error("OpenAI client not available")
            raise ImportError("openai package is required")
        
        # Audio settings optimized for Pi Zero
        self.sample_rate = 16000  # 16kHz for NVIDIA models and Pi Zero efficiency
        self.audio_format = 'wav'
        self.rpi_audio_file = "user_audio.wav"
        self.audio_device = None  # Will be set during initialization
        
        # LLM settings - can be overridden via environment variable
        # For faster voice responses, use non-reasoning models like:
        # - gpt-4o-mini, gpt-4o
        # - llama-3.1-70b-instruct, llama-3.1-8b-instruct
        # - mistral-large, mixtral-8x7b
        self.llm_model = os.getenv('VOICEBOT_LLM_MODEL', 'gpt-oss-120b')
        
        # Max tokens for LLM response
        # Reasoning models need more tokens (500+), regular models can use less (150-300)
        self.llm_max_tokens = int(os.getenv('VOICEBOT_MAX_TOKENS', '500'))
        
        # TTS voice
        self.tts_voice = "Magpie-Multilingual.EN-US.Aria"
        
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
        
        logger.info("SimpleVoiceBotRPi initialized successfully with NVIDIA models")
    
    def speech_to_text(self, audio_file_path: str) -> str:
        """Convert speech to text using NVIDIA Parakeet CTC 1.1B ASR model."""
        try:
            print("🎤 Transcribing audio with NVIDIA Parakeet...")
            update_led('BREATHE', Color.GREEN, 0.5)  # Processing
            
            # Configure ASR
            config = riva.RecognitionConfig(
                encoding=riva.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=self.sample_rate,
                language_code='en-US',
                max_alternatives=1,
                enable_automatic_punctuation=True,
                audio_channel_count=1
            )
            
            # Read audio file
            with open(audio_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Perform offline recognition
            response = self.asr_service.offline_recognize(audio_data, config)
            
            if response.results and len(response.results) > 0:
                transcription = response.results[0].alternatives[0].transcript
                print(f"✅ Transcription: {transcription}")
                return transcription
            else:
                logger.warning("No transcription results from ASR")
                return ""
                
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}", exc_info=True)
            update_led('BLINK', Color.RED, 0.75)  # Error indicator
            time.sleep(2)
            return ""
    
    def generate_response(self, user_text: str) -> str:
        """Generate response using LiteLLM."""
        try:
            print("🤖 Generating response with LiteLLM...")
            update_led('BREATHE', Color.BLUE, 0.5)  # Thinking
            
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.7,
                max_tokens=self.llm_max_tokens
            )
            
            message = response.choices[0].message
            bot_reply = message.content
            
            # For reasoning models (like o1), check reasoning_content if content is empty
            if not bot_reply or bot_reply.strip() == "":
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    logger.info("Using reasoning_content from reasoning model")
                    # Extract a reasonable response from reasoning content (first few sentences)
                    reasoning = message.reasoning_content
                    # Take first 2-3 sentences for voice response
                    sentences = reasoning.split('.')[:3]
                    bot_reply = '. '.join(sentences) + '.'
                else:
                    logger.warning(f"Empty response from LLM. Full response object: {response}")
                    return "I apologize, but I didn't generate a response. Please try again."
            
            print(f"🤖 Assistant: {bot_reply}")
            return bot_reply
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            update_led('BLINK', Color.RED, 0.75)  # Error indicator
            time.sleep(2)
            return "I'm sorry, I encountered an error processing your request."
    
    def text_to_speech(self, text: str, output_file: str) -> bool:
        """Convert text to speech using NVIDIA Magpie TTS."""
        try:
            print("🔊 Synthesizing speech with NVIDIA Magpie...")
            update_led('BREATHE', Color.BLUE, 0.5)  # Synthesizing
            
            # Create TTS request
            req = {
                "text": text,
                "language_code": "en-US",
                "encoding": riva.AudioEncoding.LINEAR_PCM,
                "sample_rate_hz": self.sample_rate,
                "voice_name": self.tts_voice
            }
            
            # Synthesize speech
            response = self.tts_service.synthesize(**req)
            
            # Convert raw PCM audio bytes to numpy array and save as proper WAV file
            audio_data = np.frombuffer(response.audio, dtype=np.int16)
            sf.write(output_file, audio_data, self.sample_rate, 'PCM_16')
            
            print("✅ Speech synthesis completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}", exc_info=True)
            update_led('BLINK', Color.RED, 0.75)  # Error indicator
            time.sleep(2)
            return False
    
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
    
    def play_audio(self, audio_file_path: str):
        """Play audio from file using aplay (RPi-specific)."""
        update_led('BLINK', Color.BLUE, 0.75)  # Bot responding
        
        try:
            print("🔊 Playing response using aplay...")
            aplay_command = ['aplay', audio_file_path] 
            playback_process = subprocess.run(aplay_command, capture_output=True, text=True)
            if playback_process.returncode != 0:
                logger.error(f"aplay error: {playback_process.stderr}")
            else:
                print("✅ Audio playback completed via aplay")
            
        except Exception as e:
            logger.error(f"Error playing audio with aplay: {e}")
    
    def process_interaction(self):
        """Handles a single interaction: STT -> LLM -> TTS."""
        try:
            if not os.path.exists(self.rpi_audio_file):
                print("No audio file found.")
                return

            # Step 1: Speech to Text
            user_text = self.speech_to_text(self.rpi_audio_file)
            
            if not user_text:
                print("❌ No transcription received")
                update_led('BLINK', Color.RED, 0.5)
                time.sleep(1)
                return
            
            print(f"👤 You said: {user_text}")
            
            # Step 2: Generate Response
            bot_reply = self.generate_response(user_text)
            
            if not bot_reply:
                print("❌ No response generated")
                update_led('BLINK', Color.RED, 0.5)
                time.sleep(1)
                return
            
            # Step 3: Text to Speech
            temp_output_path = "temp_response.wav"
            success = self.text_to_speech(bot_reply, temp_output_path)
            
            if success and os.path.exists(temp_output_path):
                # Step 4: Play Audio
                self.play_audio(temp_output_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_output_path)
                except OSError as e:
                    logger.error(f"Error removing temp audio file: {e}")
            else:
                print("❌ Failed to synthesize speech")
                update_led('BLINK', Color.RED, 0.5)
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error during interaction: {e}", exc_info=True)
            update_led('BLINK', Color.RED, 0.75)
            time.sleep(2)

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
                '-r', '16000',  # 16kHz sample rate
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
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                # Log device test failure and continue to next device
                print(f"Failed to test device {device}: {e}")
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
            '-r', '16000',  # 16kHz sample rate
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
    if not RIVA_AVAILABLE:
        missing_requirements.append("nvidia-riva-client")
    else:
        print("✓ nvidia-riva-client available")
    
    if not OPENAI_AVAILABLE:
        missing_requirements.append("openai")
    else:
        print("✓ openai package available")
    
    try:
        import yaml
        print("✓ pyyaml available")
    except ImportError:
        missing_requirements.append("pyyaml")
    
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
        if "nvidia-riva-client" in missing_requirements:
            print("  pip install nvidia-riva-client")
        if "openai" in missing_requirements:
            print("  pip install openai")
        if "pyyaml" in missing_requirements:
            print("  pip install pyyaml")
        if "alsa-utils" in [req for req in missing_requirements if "alsa-utils" in req]:
            print("  sudo apt update && sudo apt install alsa-utils")
        return False
    
    print("✓ All requirements satisfied")
    return True

def main():
    global leds
    
    print("=== RPi Voice Bot with NVIDIA Models Startup ===")
    
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
    try:
        bot = SimpleVoiceBotRPi()
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        print("❌ Bot initialization failed. Please check your API keys and configuration.")
        return
    
    if IS_RASPBERRY_PI and GPIO_AVAILABLE:
        update_led('ON', Color.CYAN, 0.1)  # Idle
        
        print("Raspberry Pi VoiceBot activated with NVIDIA models. Press button to start/stop recording.")
        
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
                                    '-r', '16000',  # 16kHz sample rate for NVIDIA models
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
                                if os.path.exists(bot.rpi_audio_file):
                                    bot.process_interaction()
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
                        except (ChildProcessError, OSError, subprocess.TimeoutExpired) as e:
                            print(f"arecord process killed (wait failed or already dead): {type(e).__name__}: {e}")
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
