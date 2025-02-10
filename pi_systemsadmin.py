import subprocess
import time
from helper_functions.chat_generation_with_internet import internet_connected_chatbot

def run_diagnostic_command(command, description):
    try:
        result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        return f"=== {description} ===\n{result.strip()}\n"
    except subprocess.CalledProcessError as e:
        return f"=== {description} ===\nError: {e.output}\n"

def get_system_diagnostics():
    diagnostics = []
    
    # System Resources
    diagnostics.append(run_diagnostic_command("top -bn1 | head -n 5", "CPU/Memory Summary"))
    diagnostics.append(run_diagnostic_command("free -h", "Memory Details"))
    diagnostics.append(run_diagnostic_command("df -h", "Disk Usage"))
    
    # Process Analysis
    diagnostics.append(run_diagnostic_command("ps aux --sort=-%mem | head -n 10", "Top Memory Processes"))
    diagnostics.append(run_diagnostic_command("ps aux --sort=-%cpu | head -n 10", "Top CPU Processes"))
    
    # Hardware Health
    diagnostics.append(run_diagnostic_command("vcgencmd measure_temp", "CPU Temperature"))
    diagnostics.append(run_diagnostic_command("vcgencmd get_throttled", "Throttling Status"))
    
    # System Logs
    diagnostics.append(run_diagnostic_command("dmesg | tail -n 20", "Recent Kernel Logs"))
    diagnostics.append(run_diagnostic_command("journalctl -p 3 -xb --no-pager", "System Errors"))
    
    return "\n".join(diagnostics)

def analyze_with_llm(diagnostics):
    system_prompt = [{
        "role": "system",
        "content": """You are a senior system administrator analyzing Raspberry Pi performance issues. 
        Provide:
        1. Summary of key issues found
        2. Top 3 resource hogs
        3. Specific suggestions to resolve slowdowns
        4. Any critical errors needing immediate attention
        Format response with clear headings and bullet points."""
    }]
    
    user_query = f"Raspberry Pi Performance Diagnostics:\n{diagnostics}"
    
    return internet_connected_chatbot(
        user_query, 
        system_prompt, 
        model_name="GEMINI",  # or "GROQ"
        max_tokens=1500,
        temperature=0.3
    )

if __name__ == '__main__':
    print("🔍 Running comprehensive system diagnostics...")
    start_time = time.time()
    
    # Collect diagnostics
    diagnostics = get_system_diagnostics()
    
    # Save raw output
    with open("pi_diagnostics.log", "w") as f:
        f.write(diagnostics)
    
    print(f"📊 Collected {len(diagnostics.splitlines())} lines of diagnostic data")
    
    # Get LLM analysis
    print("\n🤖 Analyzing with LLM...")
    analysis = analyze_with_llm(diagnostics)
    
    # Display results
    print("\n📝 Performance Report:")
    print(analysis)
    
    print(f"\n🕒 Total time: {time.time() - start_time:.1f} seconds")