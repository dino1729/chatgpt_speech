from calendar import c
import random
import subprocess
from helper_functions.chat_generation import generate_chat

# System spec analysis constants
SPEC_TRIGGER = "system_report"  # Custom command to trigger spec analysis
MAX_SPEC_TOKENS = 1000  # Reduced tokens for spec analysis

def get_system_info():
    """Collect hardware specs using shell commands"""
    commands = {
        "CPU": "lscpu",
        "Memory": "free -h",
        "Storage": "df -h /",
        "Model": "cat /proc/device-tree/model",
        "Temperature": "vcgencmd measure_temp"
    }
    return "\n".join([f"=== {k} ===\n{subprocess.getoutput(v)}" for k,v in commands.items()])

if __name__ == '__main__':
    system_prompt = [{
        "role": "system",
        "content": """You are a multi-mode assistant with two capabilities:
        1. General queries: Provide helpful, concise answers to normal questions
        2. System analysis: When receiving hardware specs (marked with [SYSTEM-REPORT]),
           analyze CPU/RAM/storage and assess capabilities for AI/development tasks"""
    }]
    
    conversation = system_prompt.copy()
    temperature = 0.5
    max_tokens = 4840

    while True:
        user_query = input("\nEnter your query: ").strip()
        
        # Handle system spec analysis
        if user_query.lower() == SPEC_TRIGGER:
            print("🛠 Collecting system specs...")
            specs = get_system_info()
            user_query = f"[SYSTEM-REPORT]\n{specs}"
            print(f"📊 Sending {len(specs.split())} words of spec data...")
            max_tokens = MAX_SPEC_TOKENS  # Temporary adjustment
        else:
            max_tokens = 4840  # Reset to default

        model_name = random.choice(["GROQ", "GEMINI", "GPT4", "GEMINI_THINKING"])
        print(f"Model: {model_name}")

        conversation.append({"role": "user", "content": user_query})
        
        assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        
        print("Bot:", assistant_reply)
        conversation.append({"role": "assistant", "content": assistant_reply})
