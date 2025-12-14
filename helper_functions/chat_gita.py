"""Bhagavad Gita Q&A assistant using OpenAI-compatible API."""
from helper_functions.chat_generation import generate_chat
from helper_functions.openai_compat import get_openai_client, get_embedding_model
from config import config
import pinecone
import json

pinecone_api_key = config.pinecone_api_key
pinecone_environment = config.pinecone_environment


def extract_context_frompinecone(query):
    """
    Extract relevant context from Pinecone vector database for a given query.
    
    Args:
        query: User's question about the Bhagavad Gita
    
    Returns:
        str: Relevant context from the Gita verses
    """
    client = get_openai_client()
    embedding_model = get_embedding_model()
    
    holybook = "gita"
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index = pinecone.Index(holybook)
    
    # Get embeddings for the query
    try:
        response = client.embeddings.create(
            input=[query], 
            model=embedding_model,
        )
        embedding = response.data[0].embedding
        
        # Find context in pinecone
        with open(f"./holybook/{holybook}.json", "r") as f:
            data = json.loads(f.read())
        res = index.query(vector=(embedding), top_k=8)
        ids = [i["id"] for i in res["matches"]]
        context = ""
        for id in ids:
            context = context + str(id) + ": " + data[str(id)] + "\n\n"
    except Exception as e:
        print("Error occurred while generating context:", str(e))
        context = "Error occurred while extracting context from Pinecone. Answer the question without context."

    return context


def gita_answer(query, history, model_name, max_tokens, temperature):
    """
    Answer questions about the Bhagavad Gita using context from vector database.
    
    Args:
        query: User's question
        history: Conversation history
        model_name: Name of the model to use for generation
        max_tokens: Maximum tokens in response
        temperature: Response creativity (0-1)
    
    Returns:
        str: Assistant's response based on Gita context
    """
    systemprompt = [{
        "role": "system",
        "content": "You are not an AI Language model. You will be a Bhagwad Gita assistant to the user. Restrict yourself to the context of the question."
    }]
    
    assistant_reply = ""
    try:
        # Set the initial conversation to the default system prompt
        conversation = systemprompt.copy()
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            conversation.append({"role": "assistant", "content": assistant})
        context = extract_context_frompinecone(query)
        userprompt = f"Here are some verses that could help answer my question:\n\n{context}\n\nMy question: {query}\n\nYour answer:\n\n"
        conversation.append({"role": "user", "content": userprompt})
        try:
            # Generate a response using the selected model
            assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        except Exception as e:
            print("Model error:", str(e))
            print("Resetting conversation...")
            conversation = systemprompt.copy()
    except Exception as e:
        print("Error occurred while generating response:", str(e))
        conversation = systemprompt.copy()

    return assistant_reply
