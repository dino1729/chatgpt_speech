"""
Chat generation module supporting multiple LLM providers.

Supports OpenAI-compatible APIs (LiteLLM/Ollama), Cohere, Gemini, and Groq.
"""
import cohere
import google.generativeai as palm
import google.generativeai as genai
from groq import Groq
from helper_functions.openai_compat import get_openai_client, get_chat_model
from config import config

cohere_api_key = config.cohere_api_key
cohere_model = config.cohere_model
google_api_key = config.google_api_key
gemini_model_name = config.gemini_model_name
gemini_thinkingmodel_name = config.gemini_thinkingmodel_name
palm_model = config.palm_model
groq_api_key = config.groq_api_key
groq_default_model = config.groq_default_model
groq_llama_model = config.groq_llama_model
groq_mixtral_model = config.groq_mixtral_model


def generate_chat(model_name, conversation, temperature, max_tokens):
    """
    Generate a chat response using the specified model.
    
    Args:
        model_name: Name of the model to use. Options:
            - "COHERE": Cohere Command-R
            - "PALM": Google PaLM
            - "GEMINI": Google Gemini
            - "GEMINI_THINKING": Google Gemini thinking model
            - "GPT4": OpenAI-compatible smart model
            - "GPT35TURBO" / "GPT4OMINI": OpenAI-compatible fast model
            - "MIXTRAL8x7B": OpenAI-compatible default model
            - "GROQ", "GROQ_LLAMA", "GROQ_MIXTRAL": Groq models
        conversation: List of message dicts with "role" and "content"
        temperature: Response creativity (0-1)
        max_tokens: Maximum tokens in response
    
    Returns:
        str: Generated response text
    """
    # Get the OpenAI-compatible client (configured for litellm or ollama)
    client = get_openai_client()
    
    if model_name == "COHERE":
        co = cohere.Client(cohere_api_key)
        response = co.chat(
            model=cohere_model,
            message=str(conversation).replace("'", '"'),
            temperature=temperature,
            max_tokens=max_tokens,
            connectors=[{"id": "web-search"}]
        )
        return response.text
    
    elif model_name == "PALM":
        palm.configure(api_key=google_api_key)
        response = palm.chat(
            model=palm_model,
            messages=str(conversation).replace("'", '"'),
            temperature=temperature,
        )
        return response.last
    
    elif model_name == "GEMINI":
        genai.configure(api_key=google_api_key)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.9,
            "top_k": 1,
        }
        gemini = genai.GenerativeModel(model_name=gemini_model_name, generation_config=generation_config)
        response = gemini.generate_content(str(conversation).replace("'", '"'))
        return response.text

    elif model_name == "GEMINI_THINKING":
        genai.configure(api_key=google_api_key)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.9,
            "top_k": 1,
        }
        gemini = genai.GenerativeModel(model_name=gemini_thinkingmodel_name, generation_config=generation_config)
        response = gemini.generate_content(str(conversation).replace("'", '"'))
        return response.text
    
    elif model_name == "GPT4":
        # Use the "smart" model from config
        response = client.chat.completions.create(
            model=get_chat_model("smart"),
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name in ("GPT35TURBO", "GPT4OMINI"):
        # Use the "fast" model from config
        response = client.chat.completions.create(
            model=get_chat_model("fast"),
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "MIXTRAL8x7B":
        # Use the default model from config
        response = client.chat.completions.create(
            model=get_chat_model("default"),
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    elif model_name == "GROQ":
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model=groq_default_model,
            messages=conversation,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content

    elif model_name == "GROQ_LLAMA":
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model=groq_llama_model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "GROQ_MIXTRAL":
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model=groq_mixtral_model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content

    else:
        return "Invalid model name"
