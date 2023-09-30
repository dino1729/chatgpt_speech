import openai
import os
import dotenv
from llama_index.agent import OpenAIAgent
from llama_index.llms import AzureOpenAI
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_hub.tools.google_search.base import GoogleSearchToolSpec
from llama_hub.tools.weather.base import OpenWeatherMapToolSpec
from llama_index import LangchainEmbedding, ServiceContext, set_global_service_context
from langchain.embeddings import OpenAIEmbeddings

# Get API keys from environment variables
dotenv.load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")
openweather_api_key = os.environ.get("OPENWEATHER_API_KEY")

os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_API_KEY")
openai.api_type = "azure"
openai.api_base = os.environ.get("AZURE_API_BASE")
openai.api_key = os.environ.get("AZURE_API_KEY")

azure_api_key = os.environ["AZURE_API_KEY"]
azure_api_type = "azure"
azure_api_base = os.environ.get("AZURE_API_BASE")
azure_api_version = os.environ.get("AZURE_API_VERSION")
azure_chatapi_version = os.environ.get("AZURE_CHATAPI_VERSION")

# Check if user set the davinci model flag
LLM_DEPLOYMENT_NAME = "gpt-4"
LLM_MODEL_NAME = "gpt-4"
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
openai.api_version = os.environ.get("AZURE_CHATAPI_FUNCTIONS_VERSION")
print("Using gpt-4 2023-07-01-preview model.")

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
    api_key=azure_api_key,
    api_base=azure_api_base,
    api_type=azure_api_type,
    api_version=azure_chatapi_version,
    temperature=0.5,
    max_tokens=1024,
)

embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model=EMBEDDINGS_DEPLOYMENT_NAME,
        deployment=EMBEDDINGS_DEPLOYMENT_NAME,
        openai_api_key=azure_api_key,
        openai_api_base=azure_api_base,
        openai_api_type=azure_api_type,
        openai_api_version=azure_api_version,
        chunk_size=16,
        max_retries=3,
    ),
    embed_batch_size=1,
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
    chunk_size=256,
)
set_global_service_context(service_context)

weather_tool = OpenWeatherMapToolSpec(
    key=openweather_api_key,
)

agent = OpenAIAgent.from_tools(
    weather_tool.to_tool_list(),
    llm=llm,
    verbose=True,
)

print("-------------------------------------------------------")
print(agent.chat("what is the current weather like in Hillsboro, Oregon"))

google_tool = GoogleSearchToolSpec(
    key=google_api_key,
    engine=google_cse_id
)

#Wrap the google search tool as it returns large payloads
tools = LoadAndSearchToolSpec.from_defaults(
    google_tool.to_tool_list()[0],
).to_tool_list()

agent = OpenAIAgent.from_tools(
    tools, 
    llm=llm,
    embedding_llm=embedding_llm,
    verbose=True
)
# agent = OpenAIAgent.from_tools(
#     google_tool.to_tool_list(), 
#     llm=llm, 
#     verbose=True
# )


print("-------------------------------------------------------")
print(agent.chat("whats the latest news summary about India?"))
print("-------------------------------------------------------")
print(agent.chat('who is Ange Postecoglou, where is he now and what is his philosophy about football?'))
print("-------------------------------------------------------")
print(agent.chat("whats the latest financial news summary?"))

