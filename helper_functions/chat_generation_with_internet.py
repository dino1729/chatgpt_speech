"""
Chat generation with internet connectivity using OpenAI-compatible LlamaIndex wrappers.

Supports Bing search, weather queries, and GPT Researcher for complex queries.
"""
import os
import requests
import asyncio
from datetime import datetime
from config import config
from helper_functions.chat_generation import generate_chat
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.weather import OpenWeatherMapToolSpec
from llama_index.tools.bing_search import BingSearchToolSpec
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaIndexOpenAIEmbedding
from newspaper import Article
from bs4 import BeautifulSoup
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, PromptHelper, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.indices import SummaryIndex
from llama_index.core import Settings
from helper_functions.researcher import get_report

# Get config values
bing_api_key = config.bing_api_key
bing_endpoint = config.bing_endpoint
bing_news_endpoint = config.bing_news_endpoint
openweather_api_key = config.openweather_api_key

sum_template = config.sum_template
ques_template = config.ques_template
summary_template = PromptTemplate(sum_template)
qa_template = PromptTemplate(ques_template)

# Get OpenAI-compatible configuration from config
openai_compat_base_url = config.openai_compat_base_url
openai_compat_api_key = config.openai_compat_api_key
openai_compat_default_model = config.openai_compat_default_model
openai_compat_embedding_model = config.openai_compat_embedding_model

temperature = config.temperature
max_tokens = config.max_tokens
model_name = config.model_name
num_output = config.num_output
max_chunk_overlap_ratio = config.max_chunk_overlap_ratio
max_input_size = config.max_input_size
context_window = config.context_window
keywords = config.keywords

# Configure LlamaIndex Settings with OpenAI-compatible LLM and embeddings
Settings.llm = LlamaIndexOpenAI(
    model=openai_compat_default_model,
    api_key=openai_compat_api_key,
    api_base=openai_compat_base_url,
)
Settings.embed_model = LlamaIndexOpenAIEmbedding(
    model=openai_compat_embedding_model,
    api_key=openai_compat_api_key,
    api_base=openai_compat_base_url,
    embed_batch_size=1,
)

text_splitter = SentenceSplitter()
Settings.text_splitter = text_splitter
Settings.prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)

BING_FOLDER = config.BING_FOLDER
if not os.path.exists(BING_FOLDER):
    os.makedirs(BING_FOLDER)

system_prompt = config.system_prompt


def saveextractedtext_to_file(text, filename):
    """Save extracted text to a file in the BING_FOLDER."""
    file_path = os.path.join(BING_FOLDER, filename)
    with open(file_path, 'w') as file:
        file.write(text)
    return f"Text saved to {file_path}"


def clearallfiles_bing():
    """Ensure the BING_FOLDER is empty."""
    for root, dir, files in os.walk(BING_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


def get_weather_data(query):
    """Get weather data using OpenWeatherMap through LlamaIndex agent."""
    weather_tool = OpenWeatherMapToolSpec(key=openweather_api_key)
    agent = OpenAIAgent.from_tools(
        weather_tool.to_tool_list(),
        llm=Settings.llm,
        verbose=False,
    )
    return str(agent.chat(query))


def text_extractor(url, debug=False):
    """Extract text from a URL using newspaper3k or BeautifulSoup."""
    if url:
        article = Article(url)
        try:
            article.download()
            article.parse()
            # Check if the article text has at least 75 words
            if len(article.text.split()) < 75:
                raise Exception("Article is too short. Probably the article is behind a paywall.")
        except Exception as e:
            if debug:
                print("Failed to download and parse article from URL using newspaper package: %s. Error: %s", url, str(e))
            # Try an alternate method using requests and beautifulsoup
            try:
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')
                article.text = soup.get_text()
            except Exception as e:
                if debug:
                    print("Failed to download article using beautifulsoup method from URL: %s. Error: %s", url, str(e))
        return article.text
    else:
        return None


def get_bing_agent(query):
    """Get Bing search results using LlamaIndex agent."""
    bing_tool = BingSearchToolSpec(
        api_key=bing_api_key,
    )

    agent = OpenAIAgent.from_tools(
        bing_tool.to_tool_list(),
        llm=Settings.llm,
        verbose=False,
    )

    return str(agent.chat(query))


def summarize(data_folder):
    """Summarize documents in a folder using LlamaIndex."""
    documents = SimpleDirectoryReader(data_folder).load_data()
    summary_index = SummaryIndex.from_documents(documents)
    retriever = summary_index.as_retriever(
        retriever_mode='default',
    )
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        summary_template=summary_template,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query("Generate a summary of the input context. Be as verbose as possible, while keeping the summary concise and to the point.")

    return response


def get_bing_news_results(query, num=5):
    """Get news results from Bing and summarize them."""
    clearallfiles_bing()
    # Construct a request
    mkt = 'en-US'
    params = {'q': query, 'mkt': mkt, 'freshness': 'Day', 'count': num}
    headers = {'Ocp-Apim-Subscription-Key': bing_api_key}
    response = requests.get(bing_news_endpoint, headers=headers, params=params)
    response_data = response.json()

    # Extract text from the urls and append them into a single text variable
    all_urls = [result['url'] for result in response_data['value']]
    all_snippets = [text_extractor(url) for url in all_urls]

    # Combine snippets with titles and article names
    combined_output = ""
    for i, (snippet, result) in enumerate(zip(all_snippets, response_data['value'])):
        title = f"Article {i + 1}: {result['name']}"
        if len(snippet.split()) >= 75:  # Check if article has at least 75 words
            combined_output += f"\n{title}\n{snippet}\n"

    # Format the results as a string and append current date and time
    output = f"Here's the scraped text from top {num} articles for: '{query}'. Current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n"
    output += combined_output

    # Save the output to a file
    saveextractedtext_to_file(output, "bing_results.txt")
    # Summarize the bing search response
    bingsummary = str(summarize(BING_FOLDER)).strip()

    return bingsummary


def simple_query(data_folder, query):
    """Query documents in a folder using LlamaIndex vector search."""
    documents = SimpleDirectoryReader(data_folder).load_data()
    vector_index = VectorStoreIndex.from_documents(documents)
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=6,
    )
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_template,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(query)

    return response


def get_bing_results(query, num=10):
    """Get web search results from Bing and answer the query."""
    clearallfiles_bing()
    # Construct a request
    mkt = 'en-US'
    params = {'q': query, 'mkt': mkt, 'count': num, 'responseFilter': ['Webpages', 'News']}
    headers = {'Ocp-Apim-Subscription-Key': bing_api_key}
    response = requests.get(bing_endpoint, headers=headers, params=params)
    response_data = response.json()

    # Extract snippets and append them into a single text variable
    all_snippets = [result['snippet'] for result in response_data['webPages']['value']]
    combined_snippets = '\n'.join(all_snippets)
    
    # Format the results as a string
    output = f"Here is the context from Bing for the query: '{query}'. Current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n"
    output += combined_snippets

    # Save the output to a file
    saveextractedtext_to_file(output, "bing_results.txt")
    # Query the results using llama-index
    answer = str(simple_query(BING_FOLDER, query)).strip()

    return answer


def internet_connected_chatbot(query, history, model_name, max_tokens, temperature, fast_response=True):
    """
    Chatbot with internet connectivity.
    
    Performs web searches for queries containing keywords, otherwise uses the LLM directly.
    """
    assistant_reply = "Sorry, I couldn't generate a response. Please try again."
    try:
        # Set the initial conversation to the default system prompt
        conversation = system_prompt.copy()
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            conversation.append({"role": "assistant", "content": assistant})
        conversation.append({"role": "user", "content": query})

        try:
            # If the query contains any of the keywords, perform a search
            if any(keyword in query.lower() for keyword in keywords):
                # If the query contains news
                if "news" in query.lower():
                    if fast_response:
                        assistant_reply = get_bing_news_results(query)
                    else:
                        assistant_reply = gpt_researcher(query)
                # If the query contains weather
                elif "weather" in query.lower():
                    assistant_reply = get_weather_data(query)
                else:
                    if fast_response:
                        assistant_reply = get_bing_results(query)
                    else:
                        assistant_reply = gpt_researcher(query)
            else:
                # Generate a response using the selected model
                assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        except Exception as e:
            print("Model error:", str(e))
            print("Resetting conversation...")
            conversation = system_prompt.copy()

    except Exception as e:
        print("Error occurred while generating response:", str(e))
        conversation = system_prompt.copy()

    return assistant_reply


def gpt_researcher(query):
    """
    Conducts research on a given query using GPT Researcher and returns the research report.
    
    Args:
        query (str): The research query to investigate
        
    Returns:
        str: The research report or error message
    """
    try:
        report_type = "research_report"
        report, _, _, _, _ = asyncio.run(get_report(query, report_type))
        return report if report else "Research failed to complete."
    except Exception as e:
        print(f"Error in GPT Researcher: {str(e)}")
        return f"Error conducting research: {str(e)}"
