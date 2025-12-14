"""
Configuration module for the chatgpt_speech project.

Reads all configuration from config/config.yml (YAML-only, no .env files).
Provides OpenAI-compatible client configuration based on llm_provider setting.
"""
import yaml
import os

# Determine the config directory relative to this file
config_dir = os.path.dirname(os.path.abspath(__file__))

# Load yaml config
with open(os.path.join(config_dir, "config.yml"), "r") as f:
    config_yaml = yaml.safe_load(f)

# =============================================================================
# OpenAI-compatible provider configuration
# =============================================================================
# Provider selection: "litellm" or "ollama"
llm_provider = config_yaml.get("llm_provider", "litellm")

# LiteLLM configuration
litellm_base_url = config_yaml.get("litellm_base_url", "http://localhost:4000/v1")
litellm_api_key = config_yaml.get("litellm_api_key", "")
litellm_fast_llm = config_yaml.get("litellm_fast_llm", "")
litellm_smart_llm = config_yaml.get("litellm_smart_llm", "")
litellm_strategic_llm = config_yaml.get("litellm_strategic_llm", "")
litellm_embedding = config_yaml.get("litellm_embedding", "text-embedding-3-large")
litellm_default_model = config_yaml.get("litellm_default_model", "")

# Ollama configuration
ollama_base_url = config_yaml.get("ollama_base_url", "http://localhost:11434/v1")
ollama_fast_llm = config_yaml.get("ollama_fast_llm", "qwen3:4b")
ollama_smart_llm = config_yaml.get("ollama_smart_llm", "deepseek-r1:latest")
ollama_strategic_llm = config_yaml.get("ollama_strategic_llm", "deepseek-r1:latest")
ollama_embedding = config_yaml.get("ollama_embedding", "nomic-embed-text")
ollama_default_model = config_yaml.get("ollama_default_model", "qwen3:4b")

# Derived OpenAI-compatible configuration based on selected provider
# These are the unified fields that all scripts should use
if llm_provider == "ollama":
    openai_compat_base_url = ollama_base_url
    openai_compat_api_key = "ollama"  # Ollama ignores API key, but OpenAI SDK requires a value
    openai_compat_default_model = ollama_default_model
    openai_compat_fast_model = ollama_fast_llm
    openai_compat_smart_model = ollama_smart_llm
    openai_compat_strategic_model = ollama_strategic_llm
    openai_compat_embedding_model = ollama_embedding
else:  # Default to litellm
    openai_compat_base_url = litellm_base_url
    openai_compat_api_key = litellm_api_key
    openai_compat_default_model = litellm_default_model
    openai_compat_fast_model = litellm_fast_llm
    openai_compat_smart_model = litellm_smart_llm
    openai_compat_strategic_model = litellm_strategic_llm
    openai_compat_embedding_model = litellm_embedding

# OpenAI Models (for specific use cases)
openai_audio_model = config_yaml.get("openai_audio_model", "gpt-4o-mini-audio-preview")
openai_tts_model = config_yaml.get("openai_tts_model", "tts-1")

# =============================================================================
# NVIDIA NIM Configuration
# =============================================================================
nvidia_api_key = config_yaml.get("nvidia_api_key", "")
nvidia_asr_function_id = config_yaml.get("nvidia_asr_function_id", "1598d209-5e27-4d3c-8079-4751568b1081")
nvidia_tts_function_id = config_yaml.get("nvidia_tts_function_id", "877104f7-e885-42b9-8de8-f6e4c6303969")
nvidia_tts_voice_name = config_yaml.get("nvidia_tts_voice_name", "Magpie-Multilingual.EN-US.Aria")

# =============================================================================
# Supabase Configuration
# =============================================================================
public_supabase_url = config_yaml.get("public_supabase_url", "")
supabase_service_role_key = config_yaml.get("supabase_service_role_key", "")

# =============================================================================
# Pinecone Configuration
# =============================================================================
pinecone_api_key = config_yaml.get("pinecone_api_key", "")
pinecone_environment = config_yaml.get("pinecone_environment", "")

# =============================================================================
# Cohere Configuration
# =============================================================================
cohere_api_key = config_yaml.get("cohere_api_key", "")
cohere_model = config_yaml.get("cohere_model", "command-r-08-2024")

# =============================================================================
# Google/Gemini Configuration
# =============================================================================
google_api_key = config_yaml.get("google_api_key", "")
gemini_model_name = config_yaml.get("gemini_model_name", "")
gemini_thinkingmodel_name = config_yaml.get("gemini_thinkingmodel_name", "")
palm_model = config_yaml.get("palm_model", "models/chat-bison-001")

# =============================================================================
# Groq Configuration
# =============================================================================
groq_api_key = config_yaml.get("groq_api_key", "")
groq_default_model = config_yaml.get("groq_default_model", "deepseek-r1-distill-llama-70b")
groq_llama_model = config_yaml.get("groq_llama_model", "llama3-70b-8192")
groq_mixtral_model = config_yaml.get("groq_mixtral_model", "mixtral-8x7b-32768")

# =============================================================================
# Bing Search Configuration
# =============================================================================
bing_api_key = config_yaml.get("bing_api_key", "")
_bing_endpoint_base = config_yaml.get("bing_endpoint", "")
bing_endpoint = _bing_endpoint_base + "/v7.0/search" if _bing_endpoint_base else ""
bing_news_endpoint = _bing_endpoint_base + "/v7.0/news/search" if _bing_endpoint_base else ""

# =============================================================================
# Azure Speech Services (for TTS/STT, separate from LLM)
# =============================================================================
azurespeechkey = config_yaml.get("azurespeechkey", "")
azurespeechregion = config_yaml.get("azurespeechregion", "")
azuretexttranslatorkey = config_yaml.get("azuretexttranslatorkey", "")

# =============================================================================
# Weather Configuration
# =============================================================================
openweather_api_key = config_yaml.get("openweather_api_key", "")
pyowm_api_key = config_yaml.get("pyowm_api_key", "")

# =============================================================================
# Email Configuration
# =============================================================================
yahoo_id = config_yaml.get("yahoo_id", "")
yahoo_app_password = config_yaml.get("yahoo_app_password", "")

# =============================================================================
# Firecrawl Configuration
# =============================================================================
firecrawl_server_url = config_yaml.get("firecrawl_server_url", "")
retriever = config_yaml.get("retriever", "firecrawl")

# =============================================================================
# Model Names List (for voicebot rotation)
# =============================================================================
model_names_list = config_yaml.get("model_names_list", ["GROQ_MIXTRAL", "GROQ_LLAMA", "GPT4OMINI", "GPT4", "GEMINI", "COHERE"])

# =============================================================================
# Path Configuration
# =============================================================================
UPLOAD_FOLDER = config_yaml.get('paths', {}).get('UPLOAD_FOLDER', './data')
WEB_SEARCH_FOLDER = config_yaml.get('paths', {}).get('WEB_SEARCH_FOLDER', './web_search_data')
BING_FOLDER = config_yaml.get('paths', {}).get('BING_FOLDER', './bing_data')
SUMMARY_FOLDER = config_yaml.get('paths', {}).get('SUMMARY_FOLDER', './data/summary_index')
VECTOR_FOLDER = config_yaml.get('paths', {}).get('VECTOR_FOLDER', './data/vector_index')

# =============================================================================
# Model Settings
# =============================================================================
temperature = config_yaml.get('settings', {}).get('temperature', 0.25)
max_tokens = config_yaml.get('settings', {}).get('max_tokens', 420)
model_name = config_yaml.get('settings', {}).get('model_name', 'GEMINI')
num_output = config_yaml.get('settings', {}).get('num_output', 1024)
max_chunk_overlap_ratio = config_yaml.get('settings', {}).get('max_chunk_overlap_ratio', 0.1)
max_input_size = config_yaml.get('settings', {}).get('max_input_size', 128000)
context_window = config_yaml.get('settings', {}).get('context_window', 128000)

# =============================================================================
# Prompts Configuration
# =============================================================================
prompts_file_path = os.path.join(config_dir, "prompts.yml")

# Load prompts.yml config
with open(prompts_file_path, "r") as f:
    prompts_config = yaml.safe_load(f)

# Accessing the templates
sum_template = prompts_config.get("sum_template", "")
eg_template = prompts_config.get("eg_template", "")
ques_template = prompts_config.get("ques_template", "")

system_prompt_content = prompts_config.get("system_prompt_content", "You are a helpful assistant.")
system_prompt = [{
    "role": "system",
    "content": system_prompt_content
}]

example_queries = prompts_config.get('example_queries', [])
keywords = prompts_config.get('keywords', [])

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    # OpenAI-compatible configuration (use these for LLM/embedding calls)
    'llm_provider',
    'openai_compat_base_url', 'openai_compat_api_key',
    'openai_compat_default_model', 'openai_compat_fast_model',
    'openai_compat_smart_model', 'openai_compat_strategic_model',
    'openai_compat_embedding_model',
    # OpenAI Models (for specific use cases)
    'openai_audio_model', 'openai_tts_model',
    # Provider-specific config (for advanced use cases)
    'litellm_base_url', 'litellm_api_key', 'litellm_fast_llm', 'litellm_smart_llm',
    'litellm_strategic_llm', 'litellm_embedding', 'litellm_default_model',
    'ollama_base_url', 'ollama_fast_llm', 'ollama_smart_llm',
    'ollama_strategic_llm', 'ollama_embedding', 'ollama_default_model',
    # NVIDIA
    'nvidia_api_key', 'nvidia_asr_function_id', 'nvidia_tts_function_id', 'nvidia_tts_voice_name',
    # Supabase
    'public_supabase_url', 'supabase_service_role_key',
    # Pinecone
    'pinecone_api_key', 'pinecone_environment',
    # Cohere
    'cohere_api_key', 'cohere_model',
    # Google/Gemini
    'google_api_key', 'gemini_model_name', 'gemini_thinkingmodel_name', 'palm_model',
    # Groq
    'groq_api_key', 'groq_default_model', 'groq_llama_model', 'groq_mixtral_model',
    # Bing
    'bing_api_key', 'bing_endpoint', 'bing_news_endpoint',
    # Azure Speech
    'azurespeechkey', 'azurespeechregion', 'azuretexttranslatorkey',
    # Weather
    'openweather_api_key', 'pyowm_api_key',
    # Email
    'yahoo_id', 'yahoo_app_password',
    # Firecrawl
    'firecrawl_server_url', 'retriever',
    # Paths
    'UPLOAD_FOLDER', 'WEB_SEARCH_FOLDER', 'BING_FOLDER', 'SUMMARY_FOLDER', 'VECTOR_FOLDER',
    # Settings
    'temperature', 'max_tokens', 'model_name', 'num_output',
    'max_chunk_overlap_ratio', 'max_input_size', 'context_window',
    # Model Names List
    'model_names_list',
    # Prompts
    'prompts_file_path', 'prompts_config',
    'sum_template', 'eg_template', 'ques_template',
    'system_prompt_content', 'system_prompt',
    'example_queries', 'keywords',
]
