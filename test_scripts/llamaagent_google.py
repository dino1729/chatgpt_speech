import openai
import os
import dotenv
from llama_index.agent import OpenAIAgent
from llama_index.llms import AzureOpenAI
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_hub.tools.google_search.base import GoogleSearchToolSpec
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding

# Get API keys from environment variables
dotenv.load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_API_KEY")
openai.api_type = "azure"
openai.api_base = os.environ.get("AZURE_API_BASE")
openai.api_key = os.environ.get("AZURE_API_KEY")

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
    openai.api_version = os.environ.get("AZURE_CHATAPI_FUNCTIONS_VERSION")
    print("Using gpt-3p5-turbo-16k 2023-07-01-preview model.")
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
)
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model=EMBEDDINGS_DEPLOYMENT_NAME,
        deployment=EMBEDDINGS_DEPLOYMENT_NAME,
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=os.environ.get("AZURE_API_VERSION"),
        chunk_size=32,
        max_retries=3,
    ),
    embed_batch_size=1,
)
google_spec = GoogleSearchToolSpec(key=google_api_key, engine=google_cse_id)

# Wrap the google search tool as it returns large payloads
tools = LoadAndSearchToolSpec.from_defaults(
    google_spec.to_tool_list()[0],
).to_tool_list()

agent = OpenAIAgent.from_tools(tools, llm=llm, embedding_llm=embedding_llm, verbose=True)

print(agent.chat('who is douglas adams and what is the meaning of life, meaning and the universe'))
print(agent.chat('what is the current weather like in Hillsboro, OR'))

