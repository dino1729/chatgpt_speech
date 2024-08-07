import cohere
import google.generativeai as genai
import google.generativeai as palm
import openai
import os
import dotenv

# Get API keys from environment variables
dotenv.load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"]
google_api_key = os.environ["GOOGLE_PALM_API_KEY"]
azure_api_key = os.environ["AZURE_API_KEY"]

def generate_chat(model_name, conversation, temperature, max_tokens):
    if model_name == "cohere":
        co = cohere.Client(cohere_api_key)
        response = co.generate(
            model='command-nightly',
            prompt=str(conversation).replace("'", '"'),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.generations[0].text
    if model_name == "cohere_chat":
        co = cohere.Client(cohere_api_key)
        response = co.chat(
            model='command-nightly',
            message=str(conversation).replace("'", '"'),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.text
    elif model_name == "palm":
        palm.configure(api_key=google_api_key)
        response = palm.chat(
            model="models/chat-bison-001",
            messages=str(conversation).replace("'", '"'),
            temperature=temperature,
        )
        return response.last
    elif model_name == "gemini":
        genai.configure(api_key=google_api_key)
        gemini = genai.GenerativeModel('gemini-pro')
        response = gemini.generate_content(str(conversation).replace("'", '"'))
        return response.text
    elif model_name == "openai":
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_API_BASE")
        openai.api_version = os.getenv("AZURE_CHATAPI_VERSION")
        openai.api_key = azure_api_key
        response = openai.ChatCompletion.create(
            engine="gpt-4o-mini",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    elif model_name == "llama2":
        openai.api_type = "open_ai"
        openai.api_base = os.getenv("LLAMA2_API_BASE")
        response = openai.ChatCompletion.create(
            model="llama2-7bchat",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    elif model_name == "gpt4all":
        openai.api_type = "open_ai"
        openai.api_base = os.getenv("LLAMA2_API_BASE")
        response = openai.ChatCompletion.create(
            model="ggml-gpt4all-j",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    elif model_name == "wizardlm":
        openai.api_type = "open_ai"
        openai.api_base = os.getenv("LLAMA2_API_BASE")
        response = openai.ChatCompletion.create(
            model="wizardlm-7b-8k-m",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    else:
        return "Invalid model name"

system_prompt = [{
    "role": "system",
    "content": "My name is Dino. You are a helpful and super-intelligent assistant, that accurately answers my queries. Be accurate, helpful, concise, and clear."
}]

temperature = 0.5
max_tokens = 105
# Set the initial conversation to the default system prompt
conversation = system_prompt.copy()
try:
    while True:

        user_text = input("Enter your question...\n")
        new_message = {"role": "user", "content": user_text}
        conversation.append(new_message)

        model_name = input("\nEnter the model name (cohere/cohere_chat/palm/gemini/openai/llama2/gpt4all/wizardlm): ")
        
        try:
            assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
            print("Bot said: {}".format(assistant_reply))
            new_assistant_message = {"role": "assistant", "content": assistant_reply}
            conversation.append(new_assistant_message)
        except Exception as e:
            print(e)
            print("Error generating chat response. Resetting conversation...")
            conversation = system_prompt.copy()


except KeyboardInterrupt:
    print("\nGoodbye!")
