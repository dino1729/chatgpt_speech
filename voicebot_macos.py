import os
import openai
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

from bs4 import BeautifulSoup
from newspaper import Article
from langchain.embeddings import OpenAIEmbeddings
from llama_index.llms import AzureOpenAI
from llama_index import (
    VectorStoreIndex,
    ListIndex,
    LangchainEmbedding,
    PromptHelper,
    Prompt,
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
    answer = str(simple_query("./data", query)).strip()

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
    bingsummary = str(summarize("./data")).strip()

    return bingsummary

def summarize(data_folder):
    
    # Initialize a document
    documents = SimpleDirectoryReader(data_folder).load_data()
    #index = VectorStoreIndex.from_documents(documents)
    list_index = ListIndex.from_documents(documents)
    # ListIndexRetriever
    retriever = list_index.as_retriever(
        retriever_mode='default',
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        text_qa_template=summary_template,
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
        elif model_name == "OPENAI":
            speech_config.speech_synthesis_voice_name = "en-US-AnaNeural"
        elif model_name == "COHERE":
            speech_config.speech_synthesis_voice_name = "en-US-SaraNeural"
        elif model_name == "BING+OPENAI":
            speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
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
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['message']['content']
    else:
        return "Invalid model name"

# Get API keys from environment variables
dotenv.load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"]
google_palm_api_key = os.environ["GOOGLE_PALM_API_KEY"]
azure_api_key = os.environ["AZURE_API_KEY"]
azurespeechkey = os.environ.get("AZURE_SPEECH_KEY")
azurespeechregion = os.environ.get("AZURE_SPEECH_REGION")
azuretexttranslatorkey = os.environ.get("AZURE_TEXT_TRANSLATOR_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_API_KEY")
openai.api_type = "azure"
openai.api_base = os.environ.get("AZURE_API_BASE")
openai.api_key = os.environ.get("AZURE_API_KEY")
LLM_DEPLOYMENT_NAME = "text-davinci-003"
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
bing_api_key = os.getenv("BING_API_KEY")
bing_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/search"
bing_news_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/news/search"

# max LLM token input size
max_input_size = 4096
num_output = 1024
max_chunk_overlap_ratio = 0.1
chunk_size = 512
context_window = 4096
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)
text_splitter = SentenceSplitter(
    separator=" ",
    chunk_size=chunk_size,
    chunk_overlap=20,
    paragraph_separator="\n\n\n"
)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

# Check if user set the davinci model flag
davincimodel_flag = False
if davincimodel_flag:
    LLM_DEPLOYMENT_NAME = "text-davinci-003"
    LLM_MODEL_NAME = "text-davinci-003"
    openai.api_version = os.environ.get("AZURE_API_VERSION")
    print("Using text-davinci-003 model.")
else:
    LLM_DEPLOYMENT_NAME = "gpt-3p5-turbo-16k"
    LLM_MODEL_NAME = "gpt-35-turbo-16k"
    openai.api_version = os.environ.get("AZURE_CHATAPI_VERSION")
    print("Using gpt-3p5-turbo-16k model.")

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
    temperature=0.5,
    max_tokens=1024,
)
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model=EMBEDDINGS_DEPLOYMENT_NAME,
        deployment=EMBEDDINGS_DEPLOYMENT_NAME,
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
        chunk_size=32,
        max_retries=3,
    ),
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
sum_template = (
    "You are a world-class text summarizer. We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on the context provided, your task is to summarize the input context while effectively conveying the main points and relevant information. The summary should be presented in the format of news headlines. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
)
summary_template = Prompt(sum_template)

ques_template = (
    "You are a world-class personal assistant. You will be provided snippets of information from the main context based on user's query. Here is the context:\n"
    "---------------------\n"
    "{context_str}\n"
    "\n---------------------\n"
    "Based on the context provided, your task is to answer the user's question to the best of your ability. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
)
qa_template = Prompt(ques_template)

'''
This script transcribes the native audio file to english language, sends this english text to GPT-3 for completion, and then translates the completed english text back to the native language and generates the audio response.
'''

system_prompt = [{
    "role": "system",
    "content": "You are a helpful and super-intelligent voice assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
}]
temperature = 0.5
max_tokens = 420

model_names = ["PALM", "OPENAI", "COHERE"]
model_index = 0
model_name = model_names[model_index]

UPLOAD_FOLDER = os.path.join(".", "data")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define a list of keywords that trigger Bing search
keywords = ["latest", "current", "recent", "update", "best", "top", "news", "weather", "summary", "previous"]

audio_path = "user_audio.wav"
tts_output_path = "bot_response.mp3"

encoding = tiktoken.get_encoding("cl100k_base")
max_token_count = 16000
max_timeout = 600
last_activity_time = time.time()

# Set the initial conversation to the default system prompt
conversation = system_prompt.copy()
print("Press the Enter key to start/stop recording...")
recording = False
try:
    while True:
        
        # Check if it's time to reset the conversation based on token count or inactivity
        if len(encoding.encode(json.dumps(conversation))) > max_token_count or time.time() - last_activity_time > max_timeout:
            conversation = system_prompt.copy()  # Reset the conversation to the default
            print("Conversation reset. Changing Model...") 
            # Increment the model index
            model_index = (model_index + 1) % len(model_names)
            # Get the current model name
            model_name = model_names[model_index]
            print("Swapped to model:", model_name)       

        # Update the last activity time
        last_activity_time = time.time()

        # Wait for the Enter key press
        input("Press 'Enter' and ask your question...\n")

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
                    # Check if the user's query contains any of the keywords
                    if any(keyword in english_text.lower() for keyword in keywords):
                        model_name = "BING+OPENAI"
                        # Check if the user's query contains the word "news"
                        if "news" in english_text.lower():
                            assistant_reply = get_bing_news_results(english_text)
                            print("{} Bot: {}".format(model_name, assistant_reply))
                            new_assistant_message = {"role": "assistant", "content": assistant_reply}
                            conversation.append(new_assistant_message)
                        else:
                            assistant_reply = get_bing_results(english_text)
                            print("{} Bot: {}".format(model_name, assistant_reply))
                            new_assistant_message = {"role": "assistant", "content": assistant_reply}
                            conversation.append(new_assistant_message)
                    else:
                        # Set the model name to the selected model
                        model_name = model_names[model_index]
                        # Generate a response using the selected model
                        assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
                        print("{} Bot: {}".format(model_name, assistant_reply))
                        new_assistant_message = {"role": "assistant", "content": assistant_reply}
                        conversation.append(new_assistant_message)
                    try:
                        translated_message = translate_text(assistant_reply, detected_audio_language)
                        text_to_speech(translated_message, tts_output_path, detected_audio_language, model_name)
                    except Exception as e:
                        print("Translation error:", str(e))
                        text_to_speech("Sorry, I couldn't answer that.", tts_output_path, "en-US", model_name)

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

