import openai
import os
import dotenv
from llama_index.agent import OpenAIAgent
from llama_index.llms import AzureOpenAI
from llama_hub.tools.bing_search.base import BingSearchToolSpec
from llama_hub.tools.azure_translate.base import AzureTranslateToolSpec

# Get API keys from environment variables
dotenv.load_dotenv()
bing_api_key = os.getenv("BING_API_KEY")
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

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
)
azurespeechkey = os.environ.get("AZURE_SPEECH_KEY")
azurespeechregion = os.environ.get("AZURE_SPEECH_REGION")
azuretexttranslatorkey = os.environ.get("AZURE_TEXT_TRANSLATOR_KEY")

translate_tool = AzureTranslateToolSpec(
    api_key=azuretexttranslatorkey,
    region=azurespeechregion,
)

agent = OpenAIAgent.from_tools(
    translate_tool.to_tool_list(),
    llm=llm,
    verbose=True,
)
print(agent.chat("Say I love you in 5 different languages"))

bing_tool = BingSearchToolSpec(
    api_key=bing_api_key,
)
agent = OpenAIAgent.from_tools(
    bing_tool.to_tool_list(),
    llm=llm,
    verbose=True,
)
print("-------------------------------------------------------")
print(agent.chat("whats the latest news about India"))
print("-------------------------------------------------------")
print(agent.chat("what is latest financial news from the USA"))
print("-------------------------------------------------------")
print(agent.chat("who is douglas adams and what is the meaning of life, meaning and the universe"))
