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

def generate_text(model_name, prompt, temperature, max_tokens):
    if model_name == "cohere":
        co = cohere.Client(cohere_api_key)
        response = co.generate(
            model='command-nightly',
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.generations[0].text
    elif model_name == "palm":
        palm.configure(api_key=google_palm_api_key)
        response = palm.generate_text(
            model="models/text-bison-001",
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return response.result
    elif model_name == "openai":
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_API_BASE")
        openai.api_version = os.getenv("AZURE_API_VERSION")
        openai.api_key = azure_api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].text
    else:
        return "Invalid model name"

# Example usage
prompt = """
You are an expert at solving word problems.

Solve the following problem:

I have three houses, each with three cats.
each cat owns 4 mittens, and a hat. Each mitten was
knit from 7m of yarn, each hat from 4m.
How much yarn was needed to make all the items?

Think about it step by step, and show your work.
"""
temperature = 0
max_tokens = 800

model_name = input("Enter the model name (cohere, palm, or openai): ")
response = generate_text(model_name, prompt, temperature, max_tokens)
print(f"{model_name.capitalize()} response:")
print(response)
