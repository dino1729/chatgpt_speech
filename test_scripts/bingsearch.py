import imp
import dotenv
import os
import requests
import json
from pprint import pprint
from bs4 import BeautifulSoup
from newspaper import Article
import tiktoken
import openai
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ListIndex, get_response_synthesizer, ServiceContext, set_global_service_context, LangchainEmbedding, Prompt
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.retrievers import VectorIndexRetriever
from llama_index.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index.query_engine import RetrieverQueryEngine

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Get API keys from environment variables
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_API_KEY")
openai.api_type = "azure"
openai.api_version = os.environ.get("AZURE_API_VERSION")
openai.api_base = os.environ.get("AZURE_API_BASE")
openai.api_key = os.environ.get("AZURE_API_KEY")
LLM_DEPLOYMENT_NAME = "text-davinci-003"
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
bing_api_key = os.getenv("BING_API_KEY")
#endpoint for regular search
bing_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/search"
#endpoint for news search
#bing_news_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/news/trendingtopics"
bing_news_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/news/search"

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_DEPLOYMENT_NAME,
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
    chunk_size=512,
)
set_global_service_context(service_context)

sum_template = (
    "You are a world-class text summarizer. We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on the information provided, your task is to summarize the input context while effectively conveying the main points and relevant information. The summary should be presented in the style of a news reader, using your own words to accurately capture the essence of the content. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "{query_str}"
)
summary_template = Prompt(sum_template)

ques_template = (
    "You are a world-class personal assistant connected to the internet. You will be provided snippets of information from the internet based on user's query. Here is the context:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on the information provided, your task is to answer the user's question to the best of your ability. You can use your own knowledge base to answer the question and only use the relavant information from the internet incase you don't have knowledge of the latest information to correctly answer user's question\n"
    "---------------------\n"
    "{query_str}"
)
qa_template = Prompt(ques_template)

def save_to_file(text, filename):
    # Create the data folder if it doesn't exist
    if not os.path.exists('./data'):
        os.makedirs('./data')

    # Save the output to the article.txt file
    file_path = os.path.join('./data', filename)
    with open(file_path, 'w') as file:
        file.write(text)

    return f"Text saved to {file_path}"

def download_art(url):

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
    output = f"Here is the short summary from Bing for the query: '{query}':\n"
    output += combined_snippets

    # Save the output to a file
    save_to_file(output, "bing_results.txt")
    # Query the results using llama-index
    answer = simple_query("./data", query)

    return answer

def get_bing_news_results(query, num=5):

    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt, 'freshness': 'Day', 'count': num }
    headers = { 'Ocp-Apim-Subscription-Key': bing_api_key }
    response = requests.get(bing_news_endpoint, headers=headers, params=params)
    response_data = response.json()  # Parse the JSON response
    #pprint(response_data)

    # Extract text from the urls and append them into a single text variable
    all_urls = [result['url'] for result in response_data['value']]
    all_snippets = [download_art(url) for url in all_urls]

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
    save_to_file(output, "bing_results.txt")
    # Summarize the bing search response
    summary = summarize("./data")

    return summary

def summarize(data_folder):
    # Initialize a document
    documents = SimpleDirectoryReader(data_folder).load_data()
    #index = VectorStoreIndex.from_documents(documents)
    index = ListIndex.from_documents(documents)
    # ListIndexRetriever
    retriever = index.as_retriever(retriever_mode='default')
    # tree summarize
    query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='tree_summarize', text_qa_template=summary_template)
    response = query_engine.query("Generate a summary of the input context. Be as verbose as possible")

    return response

def simple_query(data_folder, query):
    # Initialize a document
    documents = SimpleDirectoryReader(data_folder).load_data()
    #index = VectorStoreIndex.from_documents(documents)
    index = VectorStoreIndex.from_documents(documents)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )
    # # configure response synthesizer
    response_synthesizer = get_response_synthesizer(text_qa_template=qa_template)
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

query1 = input("What would you like to search Bing for? ")

# Check if the query contains the words news
if "news" in query1:
    bing_newsresults_summary = get_bing_news_results(query1)
    print(bing_newsresults_summary)

else:
    bing_searchresults = get_bing_results(query1)
    print(bing_searchresults)




