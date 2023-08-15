import openai
import os
from llama_index.llms import AzureOpenAI
from llama_index.tools import FunctionTool
from llama_index.agent import ReActAgent
import dotenv

# Get API keys from environment variables
dotenv.load_dotenv()
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
    print("Using gpt-3p5-turbo-16k model 2023-07-01-preview.")

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
)

# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
# initialize ReAct agent
agent = ReActAgent.from_tools(
    [multiply_tool],
    llm=llm,
    verbose=True
)

print(agent.chat("What is 2123 * 215123"))
