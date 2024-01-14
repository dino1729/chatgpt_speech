import os
import azure.cognitiveservices.speech as speechsdk
import cohere
import google.generativeai as palm
import sounddevice as sd
import soundfile as sf
import requests, uuid
import tiktoken
import time
import json
import dotenv
import RPi.GPIO as GPIO
from openai import OpenAI
from openai import AzureOpenAI as OpenAIAzure
from bs4 import BeautifulSoup
from newspaper import Article
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    PromptHelper,
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
    set_global_service_context,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.agent import OpenAIAgent
from llama_hub.tools.weather.base import OpenWeatherMapToolSpec
import mediapipe as mp
import cv2
from aiy.leds import (Leds, Pattern, PrivacyLed, RgbLeds, Color)
import random

def clearallfiles():
    # Ensure the UPLOAD_FOLDER is empty
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

def saveextractedtext_to_file(text, filename):

    # Save the output to the article.txt file
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, 'w') as file:
        file.write(text)

    return f"Text saved to {file_path}"

def text_extractor(url):

    if url:
        # Extract the article
        article = Article(url)
        try:
            article.download()
            article.parse()
            #Check if the article text has atleast 75 words
            if len(article.text.split()) < 75:
                raise Exception("Article is too short. Probably the article is behind a paywall.")
        except Exception as e:
            print("Failed to download and parse article from URL using newspaper package: %s. Error: %s", url, str(e))
            # Try an alternate method using requests and beautifulsoup
            try:
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')
                article.text = soup.get_text()
            except Exception as e:
                print("Failed to download article using beautifulsoup method from URL: %s. Error: %s", url, str(e))
        return article.text
    else:
        return None

def get_bing_results(query, num=10):

    clearallfiles()
    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt, 'count': num, 'responseFilter': ['Webpages','News'] }
    headers = { 'Ocp-Apim-Subscription-Key': bing_api_key }
    response = requests.get(bing_endpoint, headers=headers, params=params)
    response_data = response.json()  # Parse the JSON response

    # Extract snippets and append them into a single text variable
    all_snippets = [result['snippet'] for result in response_data['webPages']['value']]
    combined_snippets = '\n'.join(all_snippets)
    
    # Format the results as a string
    output = f"Here is the context from Bing for the query: '{query}':\n"
    output += combined_snippets

    # Save the output to a file
    saveextractedtext_to_file(output, "bing_results.txt")
    # Query the results using llama-index
    answer = str(simple_query(UPLOAD_FOLDER, query)).strip()

    return answer

def get_bing_news_results(query, num=5):

    clearallfiles()
    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt, 'freshness': 'Day', 'count': num }
    headers = { 'Ocp-Apim-Subscription-Key': bing_api_key }
    response = requests.get(bing_news_endpoint, headers=headers, params=params)
    response_data = response.json()  # Parse the JSON response
    #pprint(response_data)

    # Extract text from the urls and append them into a single text variable
    all_urls = [result['url'] for result in response_data['value']]
    all_snippets = [text_extractor(url) for url in all_urls]

    # Combine snippets with titles and article names
    combined_output = ""
    for i, (snippet, result) in enumerate(zip(all_snippets, response_data['value'])):
        title = f"Article {i + 1}: {result['name']}"
        if len(snippet.split()) >= 75:  # Check if article has at least 75 words
            combined_output += f"\n{title}\n{snippet}\n"

    # Format the results as a string
    output = f"Here's scraped text from top {num} articles for: '{query}':\n"
    output += combined_output

    # Save the output to a file
    saveextractedtext_to_file(output, "bing_results.txt")
    # Summarize the bing search response
    bingsummary = str(summarize(UPLOAD_FOLDER)).strip()

    return bingsummary

def get_weather_data(query):

    # Initialize OpenWeatherMapToolSpec
    weather_tool = OpenWeatherMapToolSpec(
        key=openweather_api_key,
    )

    agent = OpenAIAgent.from_tools(
        weather_tool.to_tool_list(),
        llm=llm,
        verbose=False,
    )

    return str(agent.chat(query))

def summarize(data_folder):

    # Initialize a document
    documents = SimpleDirectoryReader(data_folder).load_data()
    #index = VectorStoreIndex.from_documents(documents)
    summary_index = SummaryIndex.from_documents(documents)
    # SummaryIndexRetriever
    retriever = summary_index.as_retriever(
        retriever_mode='default',
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        summary_template=summary_template,
    )
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query("Generate a summary of the input context. Be as verbose as possible, while keeping the summary concise and to the point.")

    return response

def simple_query(data_folder, query):

    # Initialize a document
    documents = SimpleDirectoryReader(data_folder).load_data()
    #index = VectorStoreIndex.from_documents(documents)
    vector_index = VectorStoreIndex.from_documents(documents)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=6,
    )
    # # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_template,
    )
    # # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
    )
    response = query_engine.query(query)

    return response

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

def text_to_speech(text, output_path, language, model_name):
    
    speech_config = speechsdk.SpeechConfig(subscription=azurespeechkey, region=azurespeechregion)
    # Set the voice based on the language
    if language == "te-IN":
        speech_config.speech_synthesis_voice_name = "te-IN-ShrutiNeural"
    elif language == "hi-IN":
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
    else:
        # Use a default voice if the language is not specified or unsupported
        default_voice = "en-US-AriaNeural"
        if model_name == "PALM":
            speech_config.speech_synthesis_voice_name = "en-US-JaneNeural"
        elif model_name == "GPT4":
            speech_config.speech_synthesis_voice_name = "en-US-BlueNeural"
        elif model_name == "GPT35TURBO":
            speech_config.speech_synthesis_voice_name = "en-US-AnaNeural"
        elif model_name == "COHERE":
            speech_config.speech_synthesis_voice_name = "en-US-SaraNeural"
        elif model_name == "BING+OPENAI":
            speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        elif model_name == "WIZARDVICUNA7B":
            speech_config.speech_synthesis_voice_name = "en-US-AmberNeural"
        else:
            speech_config.speech_synthesis_voice_name = default_voice
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

    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
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
    
    elif model_name == "GPT4":

        response = client.chat.completions.create(
            model="gpt-4",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "GPT35TURBO":

        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "WIZARDVICUNA7B":

        local_client = OpenAI(
            api_key=llama2_api_key,
            base_url=llama2_api_base,
        )
        response = local_client.chat.completions.create(
            model="wizardvicuna7b-uncensored-hf",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    else:
        return "Invalid model name"

def reset_conversation():
    """
    Resets the conversation to the default and changes the model.
    """
    global conversation, model_index, model_name
    conversation = system_prompt.copy()
    print("Conversation reset. Changing Model...")
    model_index = (model_index + 1) % len(model_names)
    model_name = model_names[model_index]
    print("Swapped to model:", model_name)

def start_stop_recording():
    """
    Waits for a button press to start or stop recording.
    """
    global recording
    print("Press the button to start/stop recording...")
    GPIO.wait_for_edge(BUTTON_PIN, GPIO.RISING)
    time.sleep(0.2)
    recording = not recording

def detect_hand_landmarks():
    """
    Detects hand landmarks using MediaPipe Hands.
    """
    with mp_hand.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9) as hands:
        cap = cv2.VideoCapture(0)
        while True:
            ret, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lmList = []
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    myHands = results.multi_hand_landmarks[0]
                    for id, lm in enumerate(myHands.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
            fingers = []
            if len(lmList) != 0:
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                finger_counter = fingers.count(1)

                if finger_counter == 2:
                    time.sleep(0.5)
                    print("Victory sign detected!")
                    return True
            else:
                print("No hand landmarks detected")
                continue
        cap.release()
        return False

def start_recording():
    """
    Starts recording audio.
    """
    global recording_data
    print("Recording started...")
    with Leds() as leds:
        if recording:
            leds.update(Leds.rgb_on(Color.RED))
            recording_data = sd.rec(int(8 * 44100), samplerate=44100, channels=1)
        else:
            leds.update(Leds.rgb_off())
        
def stop_recording():
    """
    Stops recording audio and processes it.
    """
    global recording_data
    print("Recording stopped. Processing audio...")
    sd.stop()
    sf.write(audio_path, recording_data, 44100, 'PCM_16')

def transcribe_audio_to_text():
    """
    Transcribes Telugu/Hindi audio to English text using Azure Speech Recognition.
    """
    global english_text, detected_audio_language
    try:
        english_text, detected_audio_language = transcribe_audio(audio_path)
        print("You: {}; Language {}".format(english_text, detected_audio_language))
        new_message = {"role": "user", "content": english_text}
        conversation.append(new_message)
    except Exception as e:
        print("Transcription error:", str(e))
        return False
    return True

def generate_response():
    """
    Generates a response using the selected model.
    """
    global model_name, assistant_reply
    try:
        if any(keyword in english_text.lower() for keyword in keywords):
            model_name = "BING+OPENAI"
            if "news" in english_text.lower():
                assistant_reply = get_bing_news_results(english_text)
            elif "weather" in english_text.lower():
                assistant_reply = get_weather_data(english_text)
            else:
                assistant_reply = get_bing_results(english_text)
        else:
            model_name = model_names[model_index]
            assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        print("{} Bot: {}".format(model_name, assistant_reply))
        new_assistant_message = {"role": "assistant", "content": assistant_reply}
        conversation.append(new_assistant_message)
    except Exception as e:
        print("Model error:", str(e))
        return False
    return True

def translate_and_speak():
    """
    Translates the assistant's reply and converts it to speech.
    """
    try:
        translated_message = translate_text(assistant_reply, detected_audio_language)
        text_to_speech(translated_message, tts_output_path, detected_audio_language, model_name)
    except Exception as e:
        print("Translation error:", str(e))
        text_to_speech("Sorry, I couldn't answer that.", tts_output_path, "en-US", model_name)

def delete_audio_files():
    """
    Deletes the audio files.
    """
    try:
        os.remove(audio_path)
        os.remove(tts_output_path)
    except Exception as e:
        print("Error deleting audio files:", str(e))

# Get API key from environment variable
dotenv.load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"]
google_palm_api_key = os.environ["GOOGLE_PALM_API_KEY"]
azure_api_key = os.environ["AZURE_API_KEY"]
azure_api_type = "azure"
azure_api_base = os.environ.get("AZURE_API_BASE")
azure_embeddingapi_version = os.environ.get("AZURE_EMBEDDINGAPI_VERSION")
azure_chatapi_version = os.environ.get("AZURE_CHATAPI_VERSION")

azurespeechkey = os.environ.get("AZURE_SPEECH_KEY")
azurespeechregion = os.environ.get("AZURE_SPEECH_REGION")
azuretexttranslatorkey = os.environ.get("AZURE_TEXT_TRANSLATOR_KEY")

EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
# LocalAL API Base
llama2_api_type = "open_ai"
llama2_api_key = os.environ.get("LLAMA2_API_KEY")
llama2_api_base = os.environ.get("LLAMA2_API_BASE")

bing_api_key = os.getenv("BING_API_KEY")
bing_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/search"
bing_news_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/news/search"

openweather_api_key = os.environ.get("OPENWEATHER_API_KEY")

os.environ["OPENAI_API_KEY"] = azure_api_key

client = OpenAIAzure(
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_chatapi_version,
)
# Check if user set the davinci model flag
gpt4_flag = True
if gpt4_flag:
    LLM_DEPLOYMENT_NAME = "gpt-4"
    LLM_MODEL_NAME = "gpt-4"
    max_input_size = 128000
    context_window = 128000
    print("Using gpt4 model.")
else:
    LLM_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
    LLM_MODEL_NAME = "gpt-35-turbo-16k"
    max_input_size = 48000
    context_window = 16000
    print("Using gpt-35-turbo-16k model.")

system_prompt = [{
    "role": "system",
    "content": "You are a helpful and super-intelligent voice assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
}]
temperature = 0.5
max_tokens = 1024

model_names = ["GPT4", "GPT35TURBO", "COHERE", "PALM"]
model_index = 0
model_name = model_names[model_index]

UPLOAD_FOLDER = os.path.join(".", "data")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define a list of keywords that trigger Bing search
keywords = ["latest", "current", "recent", "update", "best", "top", "news", "weather", "summary", "previous"]
# max LLM token input size
num_output = 1024
max_chunk_overlap_ratio = 0.1
chunk_size = 256

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)
text_splitter = SentenceSplitter(
    separator=" ",
    chunk_size=chunk_size,
    chunk_overlap=20,
    paragraph_separator="\n\n\n",
    secondary_chunking_regex="[^,.;。]+[,.;。]?",
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode
)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_chatapi_version,
    temperature=0.25,
    max_tokens=num_output,
)
embedding_llm =AzureOpenAIEmbedding(
    model=EMBEDDINGS_DEPLOYMENT_NAME,
    azure_deployment=EMBEDDINGS_DEPLOYMENT_NAME,
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_embeddingapi_version,
    max_retries=3,
    embed_batch_size=1,
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper,
    chunk_size=chunk_size,
    context_window=context_window,
    node_parser=node_parser,
)
set_global_service_context(service_context)

#UPLOAD_FOLDER = './data'  # set the upload folder path
UPLOAD_FOLDER = os.path.join(".", "data")
BING_FOLDER = os.path.join(".", "bing_data")
SUMMARY_FOLDER = os.path.join(UPLOAD_FOLDER, "summary_index")
VECTOR_FOLDER = os.path.join(UPLOAD_FOLDER, "vector_index")

example_queries = [["Generate key 5 point summary"], ["What are 5 main ideas of this article?"], ["What are the key lessons learned and insights in this video?"], ["List key insights and lessons learned from the paper"], ["What are the key takeaways from this article?"]]
example_qs = []
summary = "No Summary available yet"

sum_template = (
    "You are a world-class text summarizer. We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on the context provided, your task is to summarize the input context while effectively conveying the main points and relevant information. The summary should be presented in a numbered list of at least 10 key points and takeaways, with a catchy headline at the top. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
)
summary_template = PromptTemplate(sum_template)

ques_template = (
    "You are a world-class personal assistant. You will be provided snippets of information from the main context based on user's query. Here is the context:\n"
    "---------------------\n"
    "{context_str}\n"
    "\n---------------------\n"
    "Based on the context provided, your task is to answer the user's question to the best of your ability. Try to long answers to certain questions in the form of a bulleted list. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
)
qa_template = PromptTemplate(ques_template)

'''
This script transcribes the native audio file to english language, sends this english text to GPT-3 for completion, and then translates the completed english text back to the native language and generates the audio response.
'''

# Randomly select a model_index value from 0 to 3
model_index = random.randint(0, 3)
model_name = model_names[model_index]
print("Starting with model:", model_name)

audio_path = "user_audio.wav"
tts_output_path = "bot_response.mp3"

encoding = tiktoken.get_encoding("cl100k_base")
max_token_count = 8192
max_timeout = 600
last_activity_time = time.time()

# Set the initial conversation to the default system prompt
conversation = system_prompt.copy()

recording = False

# Set up GPIO
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 23
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Define constants for the double press detection
DOUBLE_PRESS_MAX_DELAY = 0.5  # Maximum delay between presses in seconds

# Initialize state for double press detection
last_press_time = 0
double_press_detected = False

# Main loop
try:
    while True:
        
        # Check if it's time to reset the conversation based on token count or inactivity
        # ...

        # Update the last activity time
        last_activity_time = time.time()

        # Print the prompt only once at the start or after processing is done
        if not recording:
            print("Press the button to start/stop recording...")

        # Double press detection logic
        try:
            # Wait for the first button press
            channel = GPIO.wait_for_edge(BUTTON_PIN, GPIO.RISING, timeout=1000)  # Using a timeout to prevent blocking
            if channel is None:
                # Timeout occurred, no edge detected
                continue

            # Get the current time
            current_time = time.time()
            # Calculate the time difference from the last press
            time_diff = current_time - last_press_time

            # Check if this press is within the allowed time frame for a double press
            if time_diff < DOUBLE_PRESS_MAX_DELAY:
                # Double press detected
                double_press_detected = True
            else:
                # This is the first press; wait for the second press
                double_press_detected = False

            # Update the last press time
            last_press_time = current_time

            # If a double press was detected, toggle the recording state
            if double_press_detected:
                # Add a debounce delay
                time.sleep(0.2)
                # Toggle the recording state
                recording = not recording

                if recording:
                    print("Recording started...")
                    # Start recording audio
                    recording_data = sd.rec(int(8 * 44100), samplerate=44100, channels=1)
                    # Turn on the LED
                    with Leds() as leds:
                        leds.update(Leds.rgb_on(Color.RED))

                else:
                    print("Recording stopped. Processing audio...")
                    # Turn off the LED
                    with Leds() as leds:
                        leds.update(Leds.rgb_off())
                    sd.stop()
                    sf.write(audio_path, recording_data, 44100, 'PCM_16')
                    # Turn on Green LED breathing pattern to indicate processing
                    with Leds() as leds:
                        leds.pattern = Pattern.breathe(1000)
                        leds.update(Leds.rgb_pattern(Color.GREEN))
                    # Transcribe Telugu/Hindi audio to English text using Azure Speech Recognition
                    try:
                        english_text, detected_audio_language = transcribe_audio(audio_path)
                        print("You: {}; Language {}".format(english_text, detected_audio_language))
                        new_message = {"role": "user", "content": english_text}
                        conversation.append(new_message)
                    except Exception as e:
                        print("Transcription error:", str(e))
                        continue # Skip the rest of the loop and start recording again

                    # Generate a response using the selected model
                    try:
                        # Check if the user's query contains any of the keywords
                        if any(keyword in english_text.lower() for keyword in keywords):
                            model_name = "BING+OPENAI"
                            # Check if the user's query contains the word "news"
                            if "news" in english_text.lower():
                                assistant_reply = get_bing_news_results(english_text)
                            # Check if the user's query contains the word "weather"
                            elif "weather" in english_text.lower():
                                assistant_reply = get_weather_data(english_text)
                            else:
                                assistant_reply = get_bing_results(english_text)
                        else:
                            # Set the model name to the selected model
                            model_name = model_names[model_index]
                            # Generate a response using the selected model
                            assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
                        print("{} Bot: {}".format(model_name, assistant_reply))
                        new_assistant_message = {"role": "assistant", "content": assistant_reply}
                        conversation.append(new_assistant_message)
                    except Exception as e:
                        print("Model error:", str(e))
                        continue  # Skip the rest of this loop iteration

                    try:
                        translated_message = translate_text(assistant_reply, detected_audio_language)
                        text_to_speech(translated_message, tts_output_path, detected_audio_language, model_name)
                    except Exception as e:
                        print("Translation error:", str(e))
                        text_to_speech("Sorry, I couldn't answer that.", tts_output_path, "en-US", model_name)
                        continue

                    # Delete the audio files
                    try:
                        os.remove(audio_path)
                        os.remove(tts_output_path)
                    except Exception as e:
                        print("Error deleting audio files:", str(e))
                        continue

        except RuntimeError as e:
            print("RuntimeError:", str(e))
            # Handle the error (e.g., by continuing the loop or performing some recovery action)
            continue

except KeyboardInterrupt:
    print("\nScript terminated by user.")
finally:
    # Clean up GPIO
    GPIO.cleanup()
