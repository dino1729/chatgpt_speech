from hmac import new
import os
from re import T
import re
import openai
import azure.cognitiveservices.speech as speechsdk
import cohere
import google.generativeai as palm
from regex import D
import sounddevice as sd
import soundfile as sf
import requests, uuid
import tiktoken
import time
import json
import dotenv
import RPi.GPIO as GPIO

# Get API keys from environment variables
dotenv.load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"]
google_palm_api_key = os.environ["GOOGLE_PALM_API_KEY"]
azure_api_key = os.environ["AZURE_API_KEY"]
azurespeechkey = os.environ.get("AZURE_SPEECH_KEY")
azurespeechregion = os.environ.get("AZURE_SPEECH_REGION")
azuretexttranslatorkey = os.environ.get("AZURE_TEXT_TRANSLATOR_KEY")

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.5,
    "max_tokens": 420,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

'''
This script transcribes the native audio file to english language, sends this english text to GPT-3 for completion, and then translates the completed english text back to the native language and generates the audio response.
'''


#Transcribe Indian languages to English text
def transcribe_audio(audio_file):
    # Create an instance of a speech config with your subscription key and region
    # Currently the v2 endpoint is required. In a future SDK release you won't need to set it. 
    endpoint_string = "wss://{}.stt.speech.microsoft.com/speech/universal/v2".format(azurespeechregion)
    #speech_config = speechsdk.translation.SpeechTranslationConfig(subscription=azurespeechkey, endpoint=endpoint_string)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    # set up translation parameters: source language and target languages
    # Currently the v2 endpoint is required. In a future SDK release you won't need to set it. 
    #endpoint_string = "wss://{}.stt.speech.microsoft.com/speech/universal/v2".format(service_region)
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=azurespeechkey,
        endpoint=endpoint_string,
        speech_recognition_language='en-US',
        target_languages=('en','hi','te'))
    #audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)
    # Specify the AutoDetectSourceLanguageConfig, which defines the number of possible languages
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "hi-IN", "te-IN"])
    # Creates a translation recognizer using and audio file as input.
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, 
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect_source_language_config)
    result = recognizer.recognize_once()

    translated_result = format(result.translations['en'])
    detectedSrcLang = format(result.properties[speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult])

    return translated_result, detectedSrcLang

def text_to_speech(text, output_path, language):
    speech_config = speechsdk.SpeechConfig(subscription=azurespeechkey, region=azurespeechregion)
    # Set the voice based on the language
    if language == "te-IN":
        speech_config.speech_synthesis_voice_name = "te-IN-ShrutiNeural"
    elif language == "hi-IN":
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
    else:
        # Use a default voice if the language is not specified or unsupported
        speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
    # Use the default speaker as audio output and start playing the audio
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    #speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Get the audio data from the result object
        audio_data = result.audio_data  
        # Save the audio data as a WAV file
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)
            print("Speech synthesized and saved to WAV file.")

def translate_text(text, target_language):
    
    # Add your key and endpoint
    key = azuretexttranslatorkey
    endpoint = "https://api.cognitive.microsofttranslator.com"
    # location, also known as region.
    location = azurespeechregion
    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': [target_language]
    }
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{
        'text': text
    }]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']

def generate_chat(model_name, conversation, temperature, max_tokens):
    if model_name == "COHERE":
        co = cohere.Client(cohere_api_key)
        response = co.generate(
            model='command-nightly',
            prompt=str(conversation).replace("'", '"'),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.generations[0].text
    elif model_name == "PALM":
        palm.configure(api_key=google_palm_api_key)
        response = palm.chat(
            model="models/chat-bison-001",
            messages=str(conversation).replace("'", '"'),
            temperature=temperature,
        )
        return response.last
    elif model_name == "OPENAI":
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_API_BASE")
        openai.api_version = os.getenv("AZURE_CHATAPI_VERSION")
        openai.api_key = azure_api_key
        response = openai.ChatCompletion.create(
            engine="gpt-3p5-turbo-16k",
            messages=conversation,
            **OPENAI_COMPLETION_OPTIONS,
        )
        return response['choices'][0]['message']['content']
    else:
        return "Invalid model name"

system_prompt = [{
    "role": "system",
    "content": "You are a helpful and super-intelligent assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
}]
temperature = 0.5
max_tokens = 420
model_name = "OPENAI"
audio_path = "user_audio.wav"
tts_output_path = "bot_response.mp3"
# Define the encoding
encoding = tiktoken.get_encoding("cl100k_base")
# Define the maximum token count allowed
max_token_count = 16000
# Define the maximum length of time (in seconds) that the script will wait for a user input before resetting the conversation
max_timeout = 600
# Initialize the last activity time
last_activity_time = time.time()

# Set the initial conversation to the default system prompt
conversation = system_prompt.copy()
# Set up GPIO
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 23
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

recording = False
try:
    while True:

        # Check if it's time to reset the conversation based on token count or inactivity
        if len(encoding.encode(json.dumps(conversation))) > max_token_count or time.time() - last_activity_time > max_timeout:
            conversation = system_prompt.copy()  # Reset the conversation to the default
            print("Conversation reset.")        

        print("Press the button to start/stop recording...")

        # Update the last activity time
        last_activity_time = time.time()

        # Wait for the button press
        GPIO.wait_for_edge(BUTTON_PIN, GPIO.RISING)
        # Add a debounce delay
        time.sleep(0.2)
        # Toggle the recording state
        recording = not recording

        if recording:
            print("Recording started...")
            # Start recording audio
            recording_data = sd.rec(int(8 * 44100), samplerate=44100, channels=1)
        else:
            print("Recording stopped. Processing audio...")
            sd.stop()
            sf.write(audio_path, recording_data, 44100, 'PCM_16')
            # Transcribe Telugu/Hindi audio to English text using Azure Speech Recognition
            try:
                english_text, detected_audio_language = transcribe_audio(audio_path)
                print("You: {}; Language {}".format(english_text, detected_audio_language))
                new_message = {"role": "user", "content": english_text}
                conversation.append(new_message)
                # Generate a response using the selected model
                try:
                    assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
                    print("{} Bot: {}".format(model_name, assistant_reply))
                    new_assistant_message = {"role": "assistant", "content": assistant_reply}
                    conversation.append(new_assistant_message)

                    try:
                        translated_message = translate_text(assistant_reply, detected_audio_language)
                        text_to_speech(translated_message, tts_output_path, detected_audio_language)
                    except Exception as e:
                        print("Translation error:", str(e))
                        text_to_speech("Sorry, I couldn't answer that.", tts_output_path, "en-US")

                except Exception as e:
                    print("Model error:", str(e))
                    #Reset the conversation to the default
                    print("Resetting conversation...")
                    conversation = system_prompt.copy()

                # Delete the audio files
                os.remove(audio_path)
                os.remove(tts_output_path)

            except Exception as e:
                print("Transcription error:", str(e))

except KeyboardInterrupt:
    print("\nScript terminated by user.")

finally:
    # Clean up GPIO
    GPIO.cleanup()
