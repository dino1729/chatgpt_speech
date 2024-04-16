import openai
import os
import dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.azure_openai import AzureOpenAI
from llama_index.tools.bing_search import BingSearchToolSpec
from llama_index.tools.azure_translate import AzureTranslateToolSpec

# Get API keys from environment variables
dotenv.load_dotenv()
bing_api_key = os.getenv("BING_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_API_KEY")
openai.api_type = "azure"
openai.api_base = os.environ.get("AZURE_API_BASE")
openai.api_key = os.environ.get("AZURE_API_KEY")

# Check if user set the davinci model flag
LLM_DEPLOYMENT_NAME = "gpt-4"
LLM_MODEL_NAME = "gpt-4"
openai.api_version = os.environ.get("AZURE_CHATAPI_FUNCTIONS_VERSION")
print("Using gpt-4 2023-07-01-preview model.")

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
print(agent.chat("whats the latest news summary about India?"))
print("-------------------------------------------------------")
print(agent.chat('who is Ange Postecoglou, where is he now and what is his philosophy about football?'))
print("-------------------------------------------------------")
print(agent.chat("whats the latest financial news summary?"))
