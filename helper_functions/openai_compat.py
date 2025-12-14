"""
OpenAI-compatible client factory for LiteLLM/Ollama endpoints.

This module provides a centralized way to get OpenAI-compatible clients
configured from config/config.yml. All scripts should use these functions
instead of directly instantiating OpenAI clients.

Usage:
    from helper_functions.openai_compat import get_openai_client, get_chat_model, get_embedding_model

    client = get_openai_client()
    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""
from openai import OpenAI
from config import config

# Cached client instance (module-level singleton)
_cached_client = None


def get_openai_client() -> OpenAI:
    """
    Get an OpenAI-compatible client configured for the selected provider.
    
    The client is cached at module level for efficiency. Configuration is
    read from config/config.yml based on the llm_provider setting.
    
    Returns:
        OpenAI: Configured OpenAI client pointing to LiteLLM or Ollama endpoint.
    """
    global _cached_client
    if _cached_client is None:
        _cached_client = OpenAI(
            api_key=config.openai_compat_api_key,
            base_url=config.openai_compat_base_url,
        )
    return _cached_client


def get_chat_model(model_type: str = "default") -> str:
    """
    Get the chat model name for the selected provider.
    
    Args:
        model_type: One of "default", "fast", "smart", or "strategic".
                   Defaults to "default".
    
    Returns:
        str: The model name to use with chat.completions.create()
    """
    model_map = {
        "default": config.openai_compat_default_model,
        "fast": config.openai_compat_fast_model,
        "smart": config.openai_compat_smart_model,
        "strategic": config.openai_compat_strategic_model,
    }
    return model_map.get(model_type, config.openai_compat_default_model)


def get_embedding_model() -> str:
    """
    Get the embedding model name for the selected provider.
    
    Returns:
        str: The model name to use with embeddings.create()
    """
    return config.openai_compat_embedding_model


def get_provider() -> str:
    """
    Get the currently configured LLM provider.
    
    Returns:
        str: Either "litellm" or "ollama"
    """
    return config.llm_provider


def make_chat_completion(messages: list, model_type: str = "default", **kwargs) -> str:
    """
    Convenience function to make a chat completion and return the response text.
    
    Args:
        messages: List of message dicts with "role" and "content" keys.
        model_type: One of "default", "fast", "smart", or "strategic".
        **kwargs: Additional arguments passed to chat.completions.create()
                  (e.g., temperature, max_tokens, top_p)
    
    Returns:
        str: The assistant's response text.
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        model=get_chat_model(model_type),
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content


def make_embedding(texts: list[str], **kwargs) -> list[list[float]]:
    """
    Convenience function to create embeddings for a list of texts.
    
    Args:
        texts: List of strings to embed.
        **kwargs: Additional arguments passed to embeddings.create()
    
    Returns:
        list[list[float]]: List of embedding vectors.
    """
    client = get_openai_client()
    response = client.embeddings.create(
        model=get_embedding_model(),
        input=texts,
        **kwargs
    )
    return [item.embedding for item in response.data]


def make_single_embedding(text: str, **kwargs) -> list[float]:
    """
    Convenience function to create an embedding for a single text.
    
    Args:
        text: String to embed.
        **kwargs: Additional arguments passed to embeddings.create()
    
    Returns:
        list[float]: Embedding vector.
    """
    embeddings = make_embedding([text], **kwargs)
    return embeddings[0] if embeddings else []


# Export all public functions
__all__ = [
    'get_openai_client',
    'get_chat_model',
    'get_embedding_model',
    'get_provider',
    'make_chat_completion',
    'make_embedding',
    'make_single_embedding',
]

