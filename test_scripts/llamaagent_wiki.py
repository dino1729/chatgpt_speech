import openai
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.agent.openai import OpenAIAgent
from langchain.embeddings import OpenAIEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
import dotenv

dotenv.load_dotenv()
bing_api_key = os.getenv("BING_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_API_KEY")
openai.api_type = "azure"
openai.api_base = os.environ.get("AZURE_API_BASE")
openai.api_key = os.environ.get("AZURE_API_KEY")

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
tool_spec = WikipediaToolSpec()

agent = OpenAIAgent.from_tools(
    tool_spec.to_tool_list(),
    llm=llm,
    embedding_llm=embedding_llm,
    verbose=True,
)

print(agent.chat('Who is Ben Afflecks spouse?'))
print(agent.chat('What is the GDP per capita of India in 2022?'))
