import cohere
import google.generativeai as palm
import openai
import os
import dotenv

# Get API keys from environment variables
dotenv.load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"]
google_palm_api_key = os.environ["GOOGLE_PALM_API_KEY"]
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
            model='ccommand-nightly',
            message=str(conversation).replace("'", '"'),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.text
    elif model_name == "palm":
        palm.configure(api_key=google_palm_api_key)
        response = palm.chat(
            model="models/chat-bison-001",
            messages=str(conversation).replace("'", '"'),
            temperature=temperature,
        )
        return response.last
    elif model_name == "openai":
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_API_BASE")
        openai.api_version = os.getenv("AZURE_CHATAPI_VERSION")
        openai.api_key = azure_api_key
        response = openai.ChatCompletion.create(
            engine="gpt-3p5-turbo-16k",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    else:
        return "Invalid model name"

system_prompt = [{
    "role": "system",
    "content": "My name is Dino. You are a helpful, super-intelligent and fun assistant, that discusses about interesting topics."
}]

temperature = 0.2
max_tokens = 82
max_iterations = 10

# Set the initial conversation to the default system prompt
conversation = system_prompt.copy()

model_names = ["cohere", "openai"]
model_index = 0
user_role = "user"

try:
    user_text = input("Dino said: ")
    new_message = {"role": "user", "content": user_text}
    conversation.append(new_message)

    for _ in range(max_iterations):
        model_name = model_names[model_index]
        if user_role == "user":
            user_role = "assistant"
        else:
            user_role = "user"
        try:
            assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
            print("{} Bot said: {}".format(model_name.capitalize(), assistant_reply))
            new_assistant_message = {"role": user_role, "content": assistant_reply}
            conversation.append(new_assistant_message)
        except Exception as e:
            print(e)
            print("Error generating chat response. Resetting conversation...")
            conversation = system_prompt.copy()
            break

        model_index = (model_index + 1) % len(model_names)

except KeyboardInterrupt:
    print("\nConversation stopped after {} iterations.".format(max_iterations))

print("Goodbye!")
