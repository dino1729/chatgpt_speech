import os
import openai
import azure.cognitiveservices.speech as speechsdk
import sounddevice as sd
import soundfile as sf
import requests, uuid
import gradio as gr
import numpy as np

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
    # Use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
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
    # You can pass more than one object in body.
    body = [{
        'text': text
    }]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']
# defining a function to create the prompt from the system message and the messages
def create_prompt(system_message, messages):
    prompt = system_message
    message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"
    for message in messages:
        prompt += message_template.format(message['sender'], message['text'])
    prompt += "\n<|im_start|>assistant\n"
    return prompt
# defining the system message
system_message_template = "<|im_start|>system\n{}\n<|im_end|>"
system_message = system_message_template.format("You are a helpful and super-intelligent regional assistant based in Hyderabad, Telangana, that accurately answers user queries. Be accurate, helpful, concise, and clear.")
# initializing the messages list
messages = []

def record_audio():
    #print("Speak in Telugu or Hindi: Press 'Enter' to start recording, and 'q' to stop and process the audio.")
    #input("Press 'Enter' and ask your question...")
    # Start recording audio
    audio_path = "user_audio.wav"
    recording = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
    sd.wait()
    sf.write(audio_path, recording, 44100, 'PCM_16')

    return audio_path

def process_audio(audio_path):
    # Transcribe Telugu/Hindi audio to English text using Azure Speech Recognition
    english_text, detected_audio_language = transcribe_audio(audio_path)
    return english_text, detected_audio_language

def generate_response(user_text, detected_language):
    messages.append({"sender": "user", "text": user_text})
    # create the prompt for the OpenAI API
    prompt = create_prompt(system_message, messages)
    # get a response from the OpenAI API
    response = openai.Completion.create(
        engine="gpt-3p5-turbo",
        prompt= create_prompt(system_message, messages),
        **OPENAI_COMPLETION_OPTIONS,
        stop=["<|im_end|>"]
        )
    message = response.choices[0].text.strip()
    messages.append({"sender": "bot", "text": message})
    # Translate the message to Telugu/Hindi
    translated_message = translate_text(message, detected_language)
    return message, translated_message

def generate_voice_message(message, language):
    tts_output_path = "bot_response.wav"
    # Generate voice message
    text_to_speech(message, tts_output_path, language)
    return tts_output_path

# Function to create a gradio chatbot interface with audio input and output. Interface should have audio record button and a text box to display the response along with a play button to play the response audio.
def chatbot_interface():

    chat_messages = []
    # Create a gradio interface
    #gr.Interface(fn=chatbot, inputs=gr.Audio(source='microphone',type='filepath'), outputs="audio", title="Telugu/Hindi chatGPT", server_name="voicebot", server_port=7860).launch()

    def chatbot(audio_data):

        audio = open(audio_data, "rb")
        audio_path = "user_audio.wav"
        audio_data = np.frombuffer(audio.read(), dtype=np.int16)
        sf.write(audio_path, audio_data, 44100)
        # Transcribe audio to English text
        english_text, detected_audio_language = process_audio(audio_path)
        # Append user message to the chat messages list
        chat_messages.append({"sender": "user", "text": english_text})
        # Generate response
        translated_message = generate_response(english_text, detected_audio_language)
        # Append bot message to the chat messages list
        chat_messages.append({"sender": "bot", "text": translated_message[0]})
        # Generate voice message
        tts_output_path = generate_voice_message(translated_message[1], detected_audio_language)
        # Return the voice message
        return tts_output_path, chat_messages

    # Create a gradio interface
    gr.Interface(
        fn=chatbot,
        inputs=gr.inputs.Audio(source='microphone', type='filepath', label="User Audio"),
        outputs=[
            gr.outputs.Audio(type='filepath', label="Bot Response"),
            gr.outputs.JSON(label="Chat Messages")
        ]
    ).launch()


if __name__ == "__main__":
    chatbot_interface()
