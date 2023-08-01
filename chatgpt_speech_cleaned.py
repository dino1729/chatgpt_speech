import os
from re import T
import re
import openai
import azure.cognitiveservices.speech as speechsdk
from regex import D
import sounddevice as sd
import soundfile as sf
import requests, uuid

openai.api_type = "azure"
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")
azurespeechkey = os.environ.get("AZURESPEECHKEY")
azurespeechregion = os.environ.get("AZURESPEECHREGION")

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 150,
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
    key = os.environ.get("AZURETEXTTRANSLATORKEY")
    endpoint = "https://api.cognitive.microsofttranslator.com"
    # location, also known as region.
    location = os.environ.get("AZURESPEECHREGION")
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

conversation = [{
    "role": "system",
    "content": "You are a helpful and super-intelligent regional assistant based in Hyderabad, Telangana, that accurately answers user queries. Be accurate, helpful, concise, and clear."
}]

while True:
    print("Speak in Telugu or Hindi: Press 'Enter' to start recording, and 'q' to stop and process the audio.")
    input("Press 'Enter' and ask your question...")
    
    # Start recording audio
    audio_path = "user_audio.wav"
    recording = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
    sd.wait()
    sf.write(audio_path, recording, 44100, 'PCM_16')

    # Transcribe Telugu/Hindi audio to English text using Azure Speech Recognition
    english_text, detected_audio_language = transcribe_audio(audio_path)

    print("You said {} in {}".format(english_text, detected_audio_language))
    new_message = {"role": "user", "content": english_text}
    conversation.append(new_message)

    response = openai.ChatCompletion.create(
        engine="gpt-3p5-turbo-16k",
        messages= conversation,
        **OPENAI_COMPLETION_OPTIONS,
        )

    assistant_reply = response['choices'][0]['message']['content']
    print("Bot said: {}".format(assistant_reply))

    new_assistant_message = {"role": "assistant", "content": assistant_reply}
    conversation.append(new_assistant_message)

    #Translate the message to Telugu/Hindi
    translated_message = translate_text(assistant_reply, detected_audio_language)
    # Convert bot response to Telugu/Hindi audio speech
    tts_output_path = "bot_response.mp3"
    text_to_speech(translated_message, tts_output_path, detected_audio_language)

    #Delete the audio files
    os.remove(audio_path)
    os.remove(tts_output_path)
