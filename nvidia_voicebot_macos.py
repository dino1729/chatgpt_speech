import os
import time
import logging
import sys
import yaml
import sounddevice as sd
import soundfile as sf
import numpy as np

# Load configuration from config/config.yml
from config import config

# Import NVIDIA Riva client for ASR and TTS
try:
    import riva.client as riva
    RIVA_AVAILABLE = True
except ImportError:
    print("nvidia-riva-client is not installed. Please run: pip install nvidia-riva-client")
    RIVA_AVAILABLE = False
    exit(1)

# Import OpenAI for LiteLLM text generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("openai is not installed. Please run: pip install openai")
    OPENAI_AVAILABLE = False
    exit(1)

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class NvidiaVoiceBotMacOS:
    def __init__(self):
        """Initialize the macOS voicebot with NVIDIA models and LiteLLM."""
        
        # Load system prompt from config
        try:
            with open('config/prompts.yml', 'r') as f:
                prompts_config = yaml.safe_load(f)
                self.system_prompt = prompts_config.get('system_prompt_content', 
                    "You are a world-class knowledgeable AI voice assistant, Orion. "
                    "Your mission is to assist users with any questions or tasks they have on a wide range of topics. "
                    "Use your knowledge, skills, and resources to provide accurate, relevant, and helpful responses. "
                    "Please remember that you are a voice assistant and keep answers brief, concise and within 1-2 sentences, "
                    "unless it's absolutely necessary to give a longer response. Be polite, friendly, and respectful in your "
                    "interactions, and try to satisfy the user's needs as best as you can.")
        except Exception as e:
            logger.warning(f"Could not load system prompt from config: {e}")
            self.system_prompt = (
                "You are a world-class knowledgeable AI voice assistant, Orion. "
                "Your mission is to assist users with any questions or tasks they have on a wide range of topics. "
                "Use your knowledge, skills, and resources to provide accurate, relevant, and helpful responses. "
                "Please remember that you are a voice assistant and keep answers brief, concise and within 1-2 sentences, "
                "unless it's absolutely necessary to give a longer response. Be polite, friendly, and respectful in your "
                "interactions, and try to satisfy the user's needs as best as you can."
            )
        
        # Get API keys from config
        self.nvidia_api_key = config.nvidia_api_key
        self.litellm_api_key = config.openai_compat_api_key
        self.litellm_base_url = config.openai_compat_base_url
        
        if not self.nvidia_api_key:
            logger.error("nvidia_api_key not found in config/config.yml")
            raise ValueError("nvidia_api_key is required")
        
        if not self.litellm_api_key or not self.litellm_base_url:
            logger.error("OpenAI-compatible config not found in config/config.yml")
            raise ValueError("litellm_api_key/base_url or ollama config is required")
        
        # Initialize NVIDIA Riva clients
        if RIVA_AVAILABLE:
            # ASR (Speech to Text) - Parakeet CTC 1.1B
            self.asr_auth = riva.Auth(
                uri='grpc.nvcf.nvidia.com:443',
                use_ssl=True,
                metadata_args=[
                    ['function-id', config.nvidia_asr_function_id],
                    ['authorization', f'Bearer {self.nvidia_api_key}']
                ]
            )
            self.asr_service = riva.ASRService(self.asr_auth)
            
            # TTS (Text to Speech) - Magpie TTS Multilingual
            self.tts_auth = riva.Auth(
                uri='grpc.nvcf.nvidia.com:443',
                use_ssl=True,
                metadata_args=[
                    ['function-id', config.nvidia_tts_function_id],
                    ['authorization', f'Bearer {self.nvidia_api_key}']
                ]
            )
            self.tts_service = riva.SpeechSynthesisService(self.tts_auth)
            logger.info("✓ NVIDIA Riva services initialized")
        else:
            logger.error("NVIDIA Riva client not available")
            raise ImportError("nvidia-riva-client is required")
        
        # Initialize LiteLLM client
        if OPENAI_AVAILABLE:
            self.llm_client = OpenAI(
                api_key=self.litellm_api_key,
                base_url=self.litellm_base_url
            )
            logger.info("✓ LiteLLM client initialized")
        else:
            logger.error("OpenAI client not available")
            raise ImportError("openai package is required")
        
        # Audio settings optimized for efficiency
        self.sample_rate = 16000  # 16kHz for NVIDIA models
        self.channels = 1  # Mono
        self.audio_file = "user_audio.wav"
        self.response_audio_file = "bot_response.wav"
        
        # LLM settings - uses config models
        # For faster voice responses, configure fast_llm in config/config.yml
        self.llm_model = config.openai_compat_fast_model or config.openai_compat_default_model
        
        # Max tokens for LLM response
        self.llm_max_tokens = 500
        
        # TTS voice
        self.tts_voice = config.nvidia_tts_voice_name
        
        # Conversation management
        self.conversation_history = []
        self.last_activity_time = time.time()
        self.conversation_reset_interval = 600  # 10 minutes
        
        logger.info("✓ NvidiaVoiceBotMacOS initialized successfully")
        print("\n" + "="*60)
        print("🎙️  NVIDIA Voice Bot for macOS")
        print("="*60)
        print("Models:")
        print("  • Speech-to-Text: NVIDIA Parakeet CTC 1.1B")
        print(f"  • Text Generation: LiteLLM ({self.llm_model})")
        print("  • Text-to-Speech: NVIDIA Magpie TTS")
        print("="*60 + "\n")
    
    def speech_to_text(self, audio_file_path: str) -> str:
        """Convert speech to text using NVIDIA Parakeet CTC 1.1B ASR model."""
        try:
            print("🎤 Transcribing audio with NVIDIA Parakeet...")
            
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
                print(f"✓ Transcription complete")
                return transcription
            else:
                logger.warning("No transcription results from ASR")
                return ""
                
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}", exc_info=True)
            return ""
    
    def generate_response(self, user_text: str) -> str:
        """Generate response using LiteLLM."""
        try:
            print("🤖 Generating response with LiteLLM...")
            
            # Build messages with system prompt and conversation history
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_text})
            
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
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
            
            print(f"✓ Response generated")
            return bot_reply
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            return "I'm sorry, I encountered an error processing your request."
    
    def text_to_speech(self, text: str, output_file: str) -> bool:
        """Convert text to speech using NVIDIA Magpie TTS."""
        try:
            print("🔊 Synthesizing speech with NVIDIA Magpie...")
            
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
            
            print("✓ Speech synthesis complete")
            return True
            
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}", exc_info=True)
            return False
    
    def record_audio(self, duration: int = 8) -> np.ndarray:
        """Record audio from microphone."""
        print(f"🎤 Recording for up to {duration} seconds... (Press Enter to stop)")
        recording_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16'
        )
        return recording_data
    
    def stop_recording(self, recording_data: np.ndarray):
        """Stop recording and save to file."""
        print("🛑 Recording stopped. Saving audio...")
        sd.stop()
        sf.write(self.audio_file, recording_data, self.sample_rate, 'PCM_16')
        print(f"✓ Audio saved to {self.audio_file}")
    
    def play_audio(self, audio_file_path: str):
        """Play audio file."""
        try:
            print("🔊 Playing response...")
            data, fs = sf.read(audio_file_path)
            sd.play(data, fs)
            sd.wait()  # Wait until audio finishes playing
            print("✓ Playback complete")
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def process_interaction(self):
        """Process one complete interaction: record → STT → LLM → TTS → play."""
        try:
            # Check if conversation needs to be reset
            if time.time() - self.last_activity_time > self.conversation_reset_interval:
                print("\n⟳ Resetting conversation due to inactivity\n")
                self.conversation_history = []
            
            self.last_activity_time = time.time()
            
            # Record audio (non-blocking - will record until Enter is pressed again)
            recording_data = self.record_audio(duration=8)
            
            return recording_data
            
        except Exception as e:
            logger.error(f"Error in process_interaction: {e}", exc_info=True)
            return None
    
    def process_recorded_audio(self):
        """Process the recorded audio file."""
        try:
            if not os.path.exists(self.audio_file):
                print("❌ No audio file found")
                return
            
            # Step 1: Speech to Text
            print("\n" + "-"*60)
            user_text = self.speech_to_text(self.audio_file)
            
            if not user_text:
                print("❌ No transcription received")
                return
            
            print(f"👤 You said: {user_text}")
            print("-"*60)
            
            # Step 2: Generate Response
            bot_reply = self.generate_response(user_text)
            
            if not bot_reply:
                print("❌ No response generated")
                return
            
            print(f"🤖 Assistant: {bot_reply}")
            print("-"*60)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": bot_reply})
            
            # Limit conversation history to last 10 exchanges
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # Step 3: Text to Speech
            success = self.text_to_speech(bot_reply, self.response_audio_file)
            
            if not success:
                print("❌ Failed to synthesize speech")
                return
            
            # Step 4: Play Audio
            self.play_audio(self.response_audio_file)
            
            # Clean up audio files
            try:
                if os.path.exists(self.audio_file):
                    os.remove(self.audio_file)
                if os.path.exists(self.response_audio_file):
                    os.remove(self.response_audio_file)
            except OSError as e:
                logger.error(f"Error removing audio files: {e}")
            
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Error processing recorded audio: {e}", exc_info=True)
    
    def cleanup(self):
        """Clean up resources and temporary files."""
        print("\n🧹 Cleaning up...")
        try:
            if os.path.exists(self.audio_file):
                os.remove(self.audio_file)
            if os.path.exists(self.response_audio_file):
                os.remove(self.response_audio_file)
        except OSError as e:
            logger.error(f"Error during cleanup: {e}")
        print("✓ Cleanup complete")


def main():
    """Main function to run the voice bot."""
    
    # Initialize the bot
    try:
        bot = NvidiaVoiceBotMacOS()
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        print("❌ Bot initialization failed. Please check your API keys and configuration.")
        return
    
    print("💡 Instructions:")
    print("   1. Press Enter to START recording")
    print("   2. Speak your question")
    print("   3. Press Enter again to STOP recording")
    print("   4. Wait for the response")
    print("   5. Press Ctrl+C to exit")
    print("\n" + "="*60 + "\n")
    
    recording = False
    recording_data = None
    
    try:
        while True:
            # Wait for Enter key press
            input("Press 'Enter' to start/stop recording...\n")
            
            # Toggle recording state
            recording = not recording
            
            if recording:
                # Start recording
                print("\n🔴 RECORDING STARTED...")
                recording_data = bot.record_audio(duration=8)
            else:
                # Stop recording and process
                if recording_data is not None:
                    print("⏹️  RECORDING STOPPED\n")
                    bot.stop_recording(recording_data)
                    bot.process_recorded_audio()
                    recording_data = None
                else:
                    print("⚠️  No recording in progress")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Script terminated by user")
    finally:
        bot.cleanup()
        print("👋 Goodbye!\n")


if __name__ == "__main__":
    main()

