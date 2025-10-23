# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv is not installed. Please run: pip install python-dotenv")

import smtplib
import os
import time
import logging
import sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI
from pyowm import OWM
import random
import supabase
import numpy as np
import soundfile as sf
import yaml
import requests
import json
import re

# Load only the config values we need directly from YAML
config_dir = os.path.join(".", "config")
with open(os.path.join(config_dir, "config.yml"), "r") as f:
    config_yaml = yaml.safe_load(f)

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import internet-connected chatbot, but make it optional
try:
    from helper_functions.chat_generation_with_internet import internet_connected_chatbot
    INTERNET_CHATBOT_AVAILABLE = True
except (ImportError, KeyError, Exception) as e:
    INTERNET_CHATBOT_AVAILABLE = False
    logger.warning(f"⚠ internet_connected_chatbot not available - will use standard LLM for news ({type(e).__name__})")

# Import NVIDIA Riva client for TTS
try:
    import riva.client as riva
    RIVA_AVAILABLE = True
except ImportError:
    logger.warning("nvidia-riva-client is not installed. TTS will not be available.")
    RIVA_AVAILABLE = False

# Configuration
nvidia_api_key = os.getenv('NVIDIA_NIM_API_KEY')
litellm_api_key = os.getenv('LITELLM_API_KEY')
litellm_base_url = os.getenv('LITELLM_BASE_URL')
yahoo_id = config_yaml.get('yahoo_id')
yahoo_app_password = config_yaml.get('yahoo_app_password')
pyowm_api_key = config_yaml.get('pyowm_api_key')

# Supabase configuration
supabase_service_role_key = config_yaml.get('supabase_service_role_key')
public_supabase_url = config_yaml.get('public_supabase_url')

# LLM settings
llm_model = os.getenv('VOICEBOT_LLM_MODEL')
if not llm_model:
    raise ValueError("VOICEBOT_LLM_MODEL environment variable is required")
llm_max_tokens = 2048
temperature = 0.4
embedding_model = os.getenv("EMBEDDING_MODEL")
if not embedding_model:
    raise ValueError("EMBEDDING_MODEL environment variable is required")

# TTS settings
sample_rate = 16000
tts_voice = "Magpie-Multilingual.EN-US.Aria"

# Firecrawl API settings
firecrawl_base_url = os.getenv('FIRECRAWL_BASE_URL')
firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')

# Topics list
topics = [  
    "How can I be more productive?", "How to improve my communication skills?", "How to be a better leader?",  
    "How are electric vehicles less harmful to the environment?", "How can I think clearly in adverse scenarios?",  
    "What are the tenets of effective office politics?", "How to be more creative?", "How to improve my problem-solving skills?",  
    "How to be more confident?", "How to be more empathetic?", "What can I learn from Boyd, the fighter pilot who changed the art of war?",  
    "How can I seek the mentorship I want from key influential people", "How can I communicate more effectively?",  
    "Give me suggestions to reduce using filler words when communicating highly technical topics?",  
    "How to apply the best game theory concepts in getting ahead in office poilitics?", "What are some best ways to play office politics?",  
    "How to be more persuasive, assertive, influential, impactful, engaging, inspiring, motivating, captivating and convincing in my communication?",  
    "What are the top 8 ways the tit-for-tat strategy prevails in the repeated prisoner's dilemma, and how can these be applied to succeed in life and office politics?",  
    "What are Chris Voss's key strategies from *Never Split the Difference* for hostage negotiations, and how can they apply to workplace conflicts?",  
    "How can tactical empathy (e.g., labeling emotions, mirroring) improve outcomes in high-stakes negotiations?",  
    "What is the 'Accusations Audit' technique, and how does it disarm resistance in adversarial conversations?",  
    "How do calibrated questions (e.g., *How am I supposed to do that?*) shift power dynamics in negotiations?",  
    "When should you use the 'Late-Night FM DJ Voice' to de-escalate tension during disagreements?",  
    "How can anchoring bias be leveraged to set favorable terms in salary or deal negotiations?",  
    "What are 'Black Swan' tactics for uncovering hidden information in negotiations?",  
    "How can active listening techniques improve conflict resolution in team settings?",  
    "What non-verbal cues (e.g., tone, body language) most impact persuasive communication?",  
    "How can I adapt my communication style to different personality types (e.g., assertive vs. analytical)?",  
    "What storytelling frameworks make complex ideas more compelling during presentations?",  
    "How do you balance assertiveness and empathy when delivering critical feedback?",  
    "What are strategies for managing difficult conversations (e.g., layoffs, project failures) with grace?",  
    "How can Nash Equilibrium concepts guide decision-making in workplace collaborations?",  
    "What real-world scenarios mimic the 'Chicken Game,' and how should you strategize in them?",  
    "How do Schelling Points (focal points) help teams reach consensus without direct communication?",  
    "When is tit-for-tat with forgiveness more effective than strict reciprocity in office politics?",  
    "How does backward induction in game theory apply to long-term career or project planning?",  
    "What are examples of zero-sum vs. positive-sum games in corporate negotiations?",  
    "How can Bayesian reasoning improve decision-making under uncertainty (e.g., mergers, market entry)?",  
    "How can Boyd's OODA Loop (Observe, Orient, Decide, Act) improve decision-making under pressure?",  
    "What game theory principles optimize resource allocation in cross-functional teams?",  
    "How can the 'MAD' (Mutually Assured Destruction) concept deter adversarial behavior in workplaces?", 
    "How does Conway's Law ('organizations design systems that mirror their communication structures') impact the efficiency of IP or product design?",  
    "What strategies can mitigate the negative effects of Conway's Law on modularity in IP design (e.g., reusable components)?",  
    "How can teams align their structure with IP design goals to leverage Conway's Law for better outcomes?",  
    "What are real-world examples of Conway's Law leading to inefficient or efficient IP architecture in tech companies?",  
    "How does cross-functional collaboration counteract siloed IP design as predicted by Conway's Law?",  
    "Why is communication architecture critical for scalable IP design under Conway's Law?",  
    "How can organizations use Conway's Law intentionally to improve reusability and scalability of IP blocks?",  
    "What metrics assess the impact of organizational structure (Conway's Law) on IP design quality and speed?",
    "What were Steve Jobs' key leadership principles at Apple?",
    "How did Steve Jobs' product design philosophy transform consumer electronics?",
    "What can engineers learn from Steve Jobs' approach to simplicity and user experience?",
    "How did Steve Jobs balance innovation with commercial viability?",
    "What makes Elon Musk's approach to engineering challenges unique?",
    "How does Elon Musk manage multiple revolutionary companies simultaneously?",
    "What risk management strategies does Elon Musk employ in his ventures?",
    "How has Elon Musk's first principles thinking changed traditional industries?",
    "What is the significance of Jeff Bezos' Day 1 philosophy at Amazon?",
    "How did Jeff Bezos' customer obsession shape Amazon's business model?",
    "What can be learned from Jeff Bezos' approach to long-term thinking?",
    "How does Jeff Bezos' decision-making framework handle uncertainty?",
    "How did Bill Gates transition from technology leader to philanthropist?",
    "What made Bill Gates' product strategy at Microsoft so effective?",
    "How did Bill Gates foster a culture of technical excellence?",
    "What can we learn from Bill Gates' approach to global health challenges?",
    "How did geographical factors determine which societies developed advanced technologies and conquered others?",
    "What does Jared Diamond's analysis reveal about environmental determinism in human development?",
    "What political and economic conditions in Germany enabled Hitler's rise to power?",
    "How did the Nazi regime's propaganda techniques create such effective mass manipulation?",
    "How did the ancient trade networks of the Silk Roads facilitate cultural exchange and technological diffusion?",
    "What does a Silk Roads perspective teach us about geopolitical power centers throughout history?",
    "What advanced civilizations existed in pre-Columbian Americas that challenge our historical narratives?",
    "How did indigenous American societies develop sophisticated agricultural and urban systems before European contact?",
    "How did the dual revolutions (French and Industrial) fundamentally reshape European society?",
    "What economic and social factors drove the revolutionary changes across Europe from 1789-1848?",
    "What administrative innovations allowed the Ottoman Empire to successfully govern a diverse, multi-ethnic state?",
    "How did the Ottoman Empire's position between East and West influence its cultural development?",
    "What internal factors contributed most significantly to the Roman Empire's decline?",
    "How did the rise of Christianity influence the political transformation of the Roman Empire?",
    "How did the Great Migration of African Americans transform both Northern and Southern American society?",
    "What personal stories from the Great Migration reveal about systemic racism and individual resilience?",
    "What strategic and leadership lessons can be learned from Athens' and Sparta's conflict?",
    "How did democratic Athens' political system influence its military decisions during the war?",
    "What moral dilemmas faced scientists during the Manhattan Project, and how are they relevant today?",
    "How did the development of nuclear weapons transform the relationship between science and government?"
] 

# Personalities list
personalities = [
    "Chanakya", "Sun Tzu", "Machiavelli", "Leonardo da Vinci", "Socrates", "Plato", "Aristotle",
    "Confucius", "Marcus Aurelius", "Friedrich Nietzsche", "Carl Jung", "Sigmund Freud",
    "Winston Churchill", "Abraham Lincoln", "Mahatma Gandhi", "Martin Luther King Jr.", "Nelson Mandela",
    "Albert Einstein", "Isaac Newton", "Marie Curie", "Stephen Hawking", "Richard Feynman", "Nikola Tesla",
    "Galileo Galilei", "James Clerk Maxwell", "Charles Darwin",
    "Alan Turing", "Claude Shannon", "Ada Lovelace", "Grace Hopper", "Tim Berners-Lee",
    "Linus Torvalds", "Guido van Rossum", "Dennis Ritchie",
    "Bill Gates", "Steve Jobs", "Elon Musk", "Jeff Bezos", "Satya Nadella", "Tim Cook",
    "Lisa Su", "Larry Page", "Sergey Brin", "Mark Zuckerberg", "Jensen Huang",
    "Gordon Moore", "Robert Noyce", "Andy Grove"
]

# Initialize OpenAI client for LiteLLM
llm_client = OpenAI(
    api_key=litellm_api_key,
    base_url=litellm_base_url
)

# Initialize NVIDIA Riva TTS
if RIVA_AVAILABLE and nvidia_api_key:
    tts_auth = riva.Auth(
        uri='grpc.nvcf.nvidia.com:443',
        use_ssl=True,
        metadata_args=[
            ['function-id', '877104f7-e885-42b9-8de8-f6e4c6303969'],
            ['authorization', f'Bearer {nvidia_api_key}']
        ]
    )
    tts_service = riva.SpeechSynthesisService(tts_auth)
    logger.info("✓ NVIDIA Riva TTS service initialized")
else:
    tts_service = None
    logger.warning("⚠ NVIDIA Riva TTS not available")


def text_to_speech_nvidia(text: str, output_file: str, max_chars: int = 2000) -> bool:
    """Convert text to speech using NVIDIA Magpie TTS.
    
    Args:
        text: Text to convert to speech
        output_file: Output WAV file path
        max_chars: Maximum characters per chunk (default 2000 for NVIDIA Magpie)
    """
    if not tts_service:
        logger.error("TTS service not available")
        return False
    
    try:
        print(f"🔊 Synthesizing speech with NVIDIA Magpie for {output_file}...")
        
        # If text is short enough, process in one go
        if len(text) <= max_chars:
            req = {
                "text": text,
                "language_code": "en-US",
                "encoding": riva.AudioEncoding.LINEAR_PCM,
                "sample_rate_hz": sample_rate,
                "voice_name": tts_voice
            }
            response = tts_service.synthesize(**req)
            audio_data = np.frombuffer(response.audio, dtype=np.int16)
        else:
            # Split text into chunks at sentence boundaries
            print(f"  Text is {len(text)} chars, splitting into chunks...")
            
            # Split by sentences to avoid cutting words
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence would exceed max_chars, save current chunk
                if len(current_chunk) + len(sentence) + 2 > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            print(f"  Processing {len(chunks)} chunks...")
            
            # Synthesize each chunk
            audio_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
                req = {
                    "text": chunk,
                    "language_code": "en-US",
                    "encoding": riva.AudioEncoding.LINEAR_PCM,
                    "sample_rate_hz": sample_rate,
                    "voice_name": tts_voice
                }
                response = tts_service.synthesize(**req)
                chunk_audio = np.frombuffer(response.audio, dtype=np.int16)
                audio_chunks.append(chunk_audio)
            
            # Concatenate all audio chunks
            audio_data = np.concatenate(audio_chunks)
            print(f"  Combined {len(chunks)} chunks into one audio file")
        
        # Save as WAV
        sf.write(output_file, audio_data, sample_rate, 'PCM_16')
        
        # Convert to MP3
        mp3_file = output_file.replace('.wav', '.mp3')
        os.system(f'ffmpeg -i {output_file} -codec:a libmp3lame -qscale:a 2 {mp3_file} -y 2>/dev/null')
        
        print(f"✓ Speech synthesis complete: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error in text_to_speech_nvidia: {e}", exc_info=True)
        return False


def generate_gpt_response(user_message: str, system_prompt: str = None, max_tokens: int = None) -> str:
    """Generate response using LiteLLM with OpenAI API style."""
    try:
        if system_prompt is None:
            system_prompt = """
            You are EDITH, or "Even Dead, I'm The Hero," a world-class AI assistant that is designed by Tony Stark to be a powerful tool for whoever controls it. You help Dinesh in various tasks. Your response will be converted into speech and will be played on Dinesh's smart speaker. Your responses must reflect Tony's characteristic mix of confidence and humor. Start your responses with a unique, witty and engaging introduction to grab the Dinesh's attention.
            """
        
        # Use provided max_tokens or fall back to default
        if max_tokens is None:
            max_tokens = llm_max_tokens
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        message = response.choices[0].message.content
        return message
        
    except Exception as e:
        logger.error(f"Error in generate_gpt_response: {e}", exc_info=True)
        return "I apologize, but I encountered an error processing your request."


def generate_embeddings(text: str) -> list:
    """Generate embeddings using OpenAI embedding model."""
    try:
        # For NVIDIA NIM embedding models, need to specify input_type for asymmetric models
        response = llm_client.embeddings.create(
            input=[text],
            model=embedding_model,
            extra_body={"input_type": "query"}  # Use "query" for search queries, "passage" for documents
        )
        embedding = response.data[0].embedding
        
        # Truncate to 1536 dimensions to match database
        # NVIDIA embedding models produce 2048 dims, but database expects 1536
        target_dimensions = 1536
        if len(embedding) > target_dimensions:
            embedding = embedding[:target_dimensions]
        
        return embedding
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        return []


def enhance_query_with_llm(userquery: str) -> str:
    """Enhance query for better semantic search."""
    system_message = (
        "You are an expert at rephrasing and expanding user queries to maximize semantic search recall. "
        "Given a short or ambiguous user query, rewrite it as a detailed, explicit, and verbose description, "
        "including synonyms, related concepts, and clarifying context, but do not answer the question. "
        "Output only the improved query, nothing else."
    )
    
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": userquery}
        ]
        
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=messages,
            max_tokens=128,
            temperature=0.3
        )
        
        improved_query = response.choices[0].message.content.strip()
        return improved_query
    except Exception as e:
        logger.error(f"Error enhancing query: {e}", exc_info=True)
        return userquery


def get_random_topic() -> str:
    """Get a random unused topic."""
    used_topics_file = "used_topics.txt"
    
    if os.path.exists(used_topics_file):
        with open(used_topics_file, "r") as file:
            used_topics = file.read().splitlines()
    else:
        used_topics = []
    
    unused_topics = list(set(topics) - set(used_topics))
    
    if not unused_topics:
        unused_topics = topics.copy()
        used_topics = []
    
    topic = random.choice(unused_topics)
    used_topics.append(topic)
    
    with open(used_topics_file, "w") as file:
        for used_topic in used_topics:
            file.write(f"{used_topic}\n")
    
    return topic


def get_random_personality() -> str:
    """Get a random unused personality."""
    used_personalities_file = "used_personalities.txt"
    
    if os.path.exists(used_personalities_file):
        with open(used_personalities_file, "r") as file:
            used_personalities = file.read().splitlines()
    else:
        used_personalities = []
    
    unused_personalities = list(set(personalities) - set(used_personalities))
    
    if not unused_personalities:
        unused_personalities = personalities.copy()
        used_personalities = []
    
    personality = random.choice(unused_personalities)
    used_personalities.append(personality)
    
    with open(used_personalities_file, "w") as file:
        for used_personality in used_personalities:
            file.write(f"{used_personality}\n")
    
    return personality


def generate_quote(personality: str) -> str:
    """Generate a quote from the given personality."""
    quote_prompt = f"Provide a random quote from {personality} to inspire Dinesh for the day."
    
    try:
        messages = [{"role": "user", "content": quote_prompt}]
        
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=messages,
            max_tokens=100,
            temperature=0.5
        )
        
        quote = response.choices[0].message.content.strip()
        return quote
    except Exception as e:
        logger.error(f"Error generating quote: {e}", exc_info=True)
        return f"A quote from {personality}"


def generate_gpt_response_memorypalace(user_message: str) -> str:
    """Generate response for memory palace lessons."""
    sysprompt = """
    You are EDITH, or "Even Dead, I'm The Hero," a world-class AI assistant that is designed by Tony Stark to be a powerful tool for whoever controls it. You help Dinesh in various tasks. In this scenario, you are helping Dinesh recall important concepts he learned and put them in a memory palace aka, his second brain. You will be given a topic along with the semantic search results from the memory palace. You need to generate a summary or lesson learned based on the search results. You have to praise Dinesh for his efforts and encourage him to continue learning. You can also provide additional information or tips to help him understand the topic better. You are not a replacement for human intelligence, but a tool to enhance Dinesh's intelligence. You are here to help Dinesh succeed in his learning journey. You are a positive and encouraging presence in his life. You are here to support him in his quest for knowledge and growth. You are EDITH, and you are here to help Dinesh succeed. Dinesh wants to master the best of what other people have already figured out.
    
    Additionally, for each topic, provide one historical anecdote that can go back up to 10,000 years ago when human civilization started. The lesson can include a key event, discovery, mistake, and teaching from various cultures and civilizations throughout history. This will help Dinesh gain a deeper understanding of the topic by learning from the past since if one does not know history, one thinks short term; if one knows history, one thinks medium and long term..
    
    Here's a bit more about Dinesh:
    You should be a centrist politically. I reside in Hillsboro, Oregon, and I hold the position of Senior Analog Circuit Design Engineer with eight years of work experience. I am a big believer in developing Power Delivery IPs with clean interfaces and minimal maintenance. I like to work on Raspberry Pi projects and home automation in my free time. Recently, I have taken up the exciting hobby of creating LLM applications. Currently, I am engaged in the development of a fantasy premier league recommender bot that selects the most suitable players based on statistical data for a specific fixture, all while adhering to a budget. Another project that I have set my sights on is a generativeAI-based self-driving system that utilizes text prompts as sensor inputs to generate motor drive outputs, enabling the bot to control itself. The key aspect of this design lies in achieving a latency of 1000 tokens per second for the LLM token generation, which can be accomplished using a local GPU cluster. I am particularly interested in the field of physics, particularly relativity, quantum mechanics, game theory and the simulation hypothesis. I have a genuine curiosity about the interconnectedness of things and the courage to explore and advocate for interventions, even if they may not be immediately popular or obvious. My ultimate goal is to achieve success in all aspects of life and incorporate the "systems thinking" and "critical thinking" mindset into my daily routine. I aim to apply systems thinking to various situations, both professional and personal, to gain insights into different perspectives and better understand complex problems. Currently, I am captivated by the achievements of individuals like Chanakya, Nicholas Tesla, Douglas Englebart, JCR Licklider, and Vannevar Bush, and I aspire to emulate their success. I'm also super interested in learning more about game theory and how people behave in professional settings. I'm curious about the strategies that can be used to influence others and potentially advance quickly in the workplace. I'm curious about the strategies that can be used to influence others and potentially advance quickly in the workplace. So, coach me on how to deliver my presentations, communicate clearly and concisely, and how to conduct myself in front of influential people. My ultimate goal is to lead a large organization where I can create innovative technology that can benefit billions of people and improve their lives.
    """
    
    return generate_gpt_response(user_message, system_prompt=sysprompt)


def get_random_lesson() -> str:
    """Get a random lesson from memory palace."""
    try:
        topic = get_random_topic()
        try:
            improved_topic = enhance_query_with_llm(topic)
        except Exception as e:
            print(f"Warning: Could not enhance query, using original topic. Error: {str(e)}")
            improved_topic = topic
        
        try:
            topic_embedding = generate_embeddings(improved_topic)
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            topic_embedding = generate_embeddings(topic)
        
        try:
            supabase_client = supabase.Client(public_supabase_url, supabase_service_role_key)
            similarity_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            max_matches = 10
            seen_ids = set()
            results = []
            
            for threshold in similarity_thresholds:
                if len(results) >= max_matches:
                    break
                remaining = max_matches - len(results)
                response = supabase_client.rpc('mp_search', {
                    'query_embedding': topic_embedding,
                    'similarity_threshold': threshold,
                    'match_count': remaining
                }).execute()
                
                for match in response.data:
                    match_id = match.get('id') or match.get('pk') or id(match)
                    if match_id not in seen_ids:
                        results.append(match)
                        seen_ids.add(match_id)
                if len(results) >= max_matches:
                    break
            
            contents = [
                {
                    'content': match['content'],
                    'similarity': match.get('similarity', 'N/A'),
                    'metadata': {k: v for k, v in match.items() if k not in ['content', 'similarity']}
                }
                for match in results
            ]
            
            formatted_chunks = "\n\n".join(
                f"[Similarity: {c['similarity']}]\n{c['content']}" for c in contents
            )
            
            prompt = (
                f"Today's Topic: {topic}\n\n"
                f"Based on the following lessons (each with a similarity index), generate a summary or lesson learned for the topic:\n\n"
                f"{formatted_chunks}"
            )
            lesson_learned = generate_gpt_response_memorypalace(prompt)
        except Exception as e:
            print(f"Error connecting to Supabase or retrieving matches: {str(e)}")
            prompt = (
                f"Today's Topic: {topic}\n\nPlease provide insights, lessons, and historical context about this topic. "
                f"Include one historical anecdote going back up to 10,000 years of human civilization that relates to this topic."
            )
            lesson_learned = generate_gpt_response_memorypalace(prompt)
        
        return lesson_learned
    except Exception as e:
        print(f"Critical error in get_random_lesson: {str(e)}")
        return "Today I encountered some technical difficulties retrieving your lesson. However, remember that setbacks are temporary, and persistence is key to overcoming challenges. Let's continue our learning journey tomorrow with renewed enthusiasm."


def get_weather():
    """Get weather information."""
    try:
        owm = OWM(pyowm_api_key)
        mgr = owm.weather_manager()
        weather = mgr.weather_at_id(5743413).weather  # North Plains, OR
        temp = weather.temperature('celsius')['temp']
        status = weather.detailed_status
        return temp, status
    except Exception as e:
        logger.error(f"Error getting weather: {e}")
        return 20, "clear sky"


def time_left_in_year():
    """Calculate time remaining in the year."""
    today = datetime.now()
    end_of_year = datetime(today.year, 12, 31)
    days_completed = today.timetuple().tm_yday
    weeks_completed = days_completed / 7
    delta = end_of_year - today
    days_left = delta.days + 1
    weeks_left = days_left / 7
    total_days_in_year = 366 if (today.year % 4 == 0 and today.year % 100 != 0) or (today.year % 400 == 0) else 365
    percent_days_left = (days_left / total_days_in_year) * 100
    
    return days_completed, weeks_completed, days_left, weeks_left, percent_days_left


def save_message_to_file(message: str, filename: str):
    """Save message to file."""
    try:
        os.makedirs(os.path.join("bing_data", os.path.dirname(filename)), exist_ok=True)
        with open(os.path.join("bing_data", filename), 'w', encoding='utf-8') as file:
            file.write(message)
        print(f"✓ Message saved to {os.path.join('bing_data', filename)}")
    except Exception as e:
        print(f"Failed to save message to file: {e}")


def send_email(subject: str, message: str, is_html: bool = False):
    """Send email."""
    sender_email = yahoo_id
    receiver_email = "katam.dinesh@hotmail.com"
    password = yahoo_app_password
    
    email_message = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject
    
    if is_html:
        email_message.attach(MIMEText(message, "html"))
    else:
        email_message.attach(MIMEText(message, "plain"))
    
    try:
        server = smtplib.SMTP('smtp.mail.yahoo.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = email_message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"✓ Email sent successfully: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")


def generate_gpt_response_voicebot(user_message: str, max_tokens: int = 512) -> str:
    """Generate voice-friendly response."""
    sysprompt = """
    You are EDITH, speaking through a voicebot. For the voice format:
    1. Use conversational, natural speaking tone (e.g., "Today in tech news..." or "Moving on to financial markets...")
    2. Break down complex information into simple, clear sentences
    3. Use verbal transitions between topics (e.g., "Now, let's look at..." or "In other news...")
    4. Avoid technical jargon unless necessary
    5. Keep points brief and easy to follow
    6. Never mention URLs, citations, or technical markers
    7. Use natural date formats (e.g., "today" or "yesterday" instead of MM/DD/YYYY)
    8. Focus on the story and its impact rather than sources
    9. End each section with a brief overview or key takeaway
    10. Use listener-friendly phrases like "As you might have heard" or "Interestingly"
    
    CRITICAL: Your response MUST be under 2000 characters total. Keep it concise and impactful.
    """
    
    return generate_gpt_response(user_message, system_prompt=sysprompt, max_tokens=max_tokens)


def get_news_fallback(query: str) -> str:
    """Fallback news function when internet_connected_chatbot is not available."""
    system_prompt = """
    You are EDITH, a knowledgeable AI assistant. Generate a realistic news summary based on current trends and your knowledge.
    Be informative and engaging, but acknowledge that this is based on general trends rather than real-time data.
    """
    return generate_gpt_response(query, system_prompt=system_prompt)


def search_web_with_firecrawl(
    query: str, 
    limit: int = 10,
    sources: list = None,
    categories: list = None,
    tbs: str = None,
    country: str = "US",
    scrape_content: bool = True
) -> dict:
    """
    Search the web using Firecrawl API v2.
    
    Based on official documentation: https://docs.firecrawl.dev/api-reference/endpoint/search
    
    Args:
        query: Search query (supports operators like site:, intitle:, etc.)
        limit: Number of results to return (1-100)
        sources: List of sources to search. Options: ["web", "images", "news"]
        categories: List of categories to filter by. Options: ["github", "research", "pdf"]
        tbs: Time-based search parameter (e.g., "qdr:d" for past day, "qdr:w" for past week)
        country: ISO country code (e.g., "US", "GB", "IN")
        scrape_content: If True, scrape full content of results
        
    Returns:
        Dictionary with search results in format:
        {
            "success": true,
            "data": {
                "web": [...],
                "images": [...],
                "news": [...]
            }
        }
    """
    try:
        print(f"🔍 Searching with Firecrawl: '{query[:60]}...'")
        
        search_url = f"{firecrawl_base_url}/search"
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add API key to headers if available and not default
        if firecrawl_api_key and firecrawl_api_key != 'default':
            headers["Authorization"] = f"Bearer {firecrawl_api_key}"
        
        # Build payload according to API spec
        payload = {
            "query": query,
            "limit": min(max(limit, 1), 100),  # Ensure limit is between 1-100
            "country": country,
            "timeout": 60000,  # 60 seconds timeout
            "ignoreInvalidURLs": True  # Helps reduce errors
        }
        
        # Add sources if specified (defaults to ["web"] if not specified)
        if sources:
            payload["sources"] = sources
        else:
            payload["sources"] = ["web", "news"]  # Search both web and news by default
        
        # Add categories if specified
        if categories:
            payload["categories"] = [{"type": cat} for cat in categories]
        
        # Add time-based search if specified
        if tbs:
            payload["tbs"] = tbs
        
        # Add scrape options to get full content
        if scrape_content:
            payload["scrapeOptions"] = {
                "formats": ["markdown"],  # Get markdown format
                "onlyMainContent": True,   # Extract only main content
                "timeout": 30000,          # 30 second timeout per page
                "removeBase64Images": True, # Remove base64 images to reduce size
                "blockAds": True           # Block ads
            }
        
        response = requests.post(search_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            # Count results from all sources
            web_count = len(data.get('data', {}).get('web', []))
            news_count = len(data.get('data', {}).get('news', []))
            images_count = len(data.get('data', {}).get('images', []))
            
            total = web_count + news_count + images_count
            print(f"✓ Found {total} results (web: {web_count}, news: {news_count}, images: {images_count})")
            
            if data.get('warning'):
                logger.warning(f"Firecrawl warning: {data['warning']}")
            
            return data
        else:
            logger.error(f"Firecrawl search failed: {response.status_code} - {response.text}")
            return {"success": False, "data": {"web": [], "news": [], "images": []}}
            
    except Exception as e:
        logger.error(f"Error in Firecrawl search: {e}", exc_info=True)
        return {"success": False, "data": {"web": [], "news": [], "images": []}}


def get_news_with_firecrawl(
    query: str, 
    limit: int = 10,
    time_filter: str = "qdr:d",  # Past day by default
    country: str = "US"
) -> str:
    """
    Get news using Firecrawl web search and process with LLM.
    
    Based on Firecrawl v2 API: https://docs.firecrawl.dev/api-reference/endpoint/search
    
    Args:
        query: News query/topic
        limit: Number of search results
        time_filter: Time-based filter (qdr:h=hour, qdr:d=day, qdr:w=week, qdr:m=month)
        country: ISO country code
        
    Returns:
        Formatted news summary
    """
    try:
        # Search both web and news sources with time filter
        search_results = search_web_with_firecrawl(
            query=query,
            limit=limit,
            sources=["web", "news"],  # Search both web and news
            tbs=time_filter,  # Filter by time
            country=country,
            scrape_content=True  # Get full content
        )
        
        # Extract results from both web and news sources
        web_results = search_results.get('data', {}).get('web', [])
        news_results = search_results.get('data', {}).get('news', [])
        
        if not search_results.get('success', True) or (not web_results and not news_results):
            print("  No results from Firecrawl, using fallback")
            return get_news_fallback(query)
        
        # Process and format search results
        results_text = []
        result_num = 1
        
        # Process news results first (more relevant for news queries)
        for result in news_results[:limit]:
            title = result.get('title', 'No title')
            url = result.get('url', '')
            snippet = result.get('snippet', '')
            date = result.get('date', '')
            markdown_content = result.get('markdown', '')
            
            # Build content from available fields
            content_parts = []
            if snippet:
                content_parts.append(snippet)
            if markdown_content:
                content_parts.append(markdown_content[:800])
            
            content = ' '.join(content_parts) if content_parts else ''
            
            date_str = f" [{date}]" if date else ""
            results_text.append(
                f"{result_num}. [NEWS]{date_str} {title}\n"
                f"   URL: {url}\n"
                f"   {content[:600]}\n"
            )
            result_num += 1
        
        # Then process web results
        for result in web_results[:limit]:
            if result_num > limit:
                break
                
            title = result.get('title', 'No title')
            url = result.get('url', '')
            description = result.get('description', '')
            markdown_content = result.get('markdown', '')
            
            # Extract metadata if available
            metadata = result.get('metadata', {})
            
            # Build content from available fields
            content_parts = []
            if description:
                content_parts.append(description)
            if markdown_content:
                # Take first 1000 chars of markdown
                content_parts.append(markdown_content[:1000])
            
            content = ' '.join(content_parts) if content_parts else ''
            
            results_text.append(
                f"{result_num}. [WEB] {title}\n"
                f"   URL: {url}\n"
                f"   {content[:600]}\n"
            )
            result_num += 1
        
        combined_results = "\n".join(results_text)
        
        # Use LLM to process and summarize the results
        system_prompt = """
        You are EDITH, a knowledgeable AI assistant analyzing real-time web search results.
        You are given search results from both news sources and web pages about current events.
        
        Your task:
        1. Analyze these search results and create a comprehensive, well-structured news summary
        2. Focus on the most important and recent developments
        3. Cite sources by mentioning the website name or publication
        4. Maintain factual accuracy based on the provided sources
        5. Structure your response with the most critical information first
        6. If dates are provided, mention them to show timeliness
        
        Be engaging and informative while staying true to the source material.
        """
        
        prompt = f"""
        Search Query: {query}
        Time Filter: {time_filter}
        
        Search Results from Firecrawl (Web + News):
        {combined_results}
        
        Please provide a detailed, well-structured summary of the news based on these search results. 
        Include:
        - Key developments and breaking news
        - Important context and background
        - Multiple perspectives if available
        - Relevant quotes or data points
        - Source citations
        
        Structure your response clearly with proper paragraphs and make it suitable for audio narration.
        """
        
        summary = generate_gpt_response(prompt, system_prompt=system_prompt)
        
        total_sources = len(web_results) + len(news_results)
        print(f"✓ Generated news summary from {total_sources} sources ({len(news_results)} news, {len(web_results)} web)")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in get_news_with_firecrawl: {e}", exc_info=True)
        return get_news_fallback(query)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 NVIDIA Year Progress and News Reporter")
    print("="*60)
    print(f"  • LLM Model: {llm_model}")
    print(f"  • TTS: NVIDIA Magpie TTS")
    print("="*60 + "\n")
    
    # Calculate year progress
    days_completed, weeks_completed, days_left, weeks_left, percent_days_left = time_left_in_year()
    
    # Get quote and lesson
    print("📚 Generating daily quote and lesson...")
    random_personality = get_random_personality()
    quote = generate_quote(random_personality)
    lesson_learned = get_random_lesson()
    
    # Generate year progress message
    print("\n📅 Generating year progress report...")
    year_progress_message_prompt = f"""
    Here is a year progress report for {datetime.now().strftime("%B %d, %Y")}:
    
    Days completed: {days_completed}
    Weeks completed: {weeks_completed:.1f}
    Days remaining: {days_left}
    Weeks remaining: {weeks_left:.1f}
    Year Progress: {100 - percent_days_left:.1f}% completed

    Quote of the day from {random_personality}:
    {quote}

    Today's lesson:
    {lesson_learned}
    """
    
    year_progress_gpt_response = generate_gpt_response(year_progress_message_prompt)
    
    # Save year progress message
    save_message_to_file(year_progress_gpt_response, "year_progress_report.txt")
    
    # Generate audio for year progress
    print("\n🔊 Generating audio for year progress report...")
    yearprogress_audio_path = "year_progress_report.wav"
    if text_to_speech_nvidia(year_progress_gpt_response, yearprogress_audio_path):
        print("✓ Year progress audio generated")
    
    # News updates with Firecrawl
    print("\n📰 Generating news updates with Firecrawl...")
    print("    Using time filter: past day (qdr:d)")
    print("    Sources: web + news")
    print("    Full content scraping: enabled")
    
    # Technology news
    technews_query = f"technology news latest developments AI artificial intelligence"
    print("\n📱 Technology News:")
    news_update_tech = get_news_with_firecrawl(
        query=technews_query,
        limit=10,
        time_filter="qdr:d",  # Past day
        country="US"
    )
    save_message_to_file(news_update_tech, "news_tech_report.txt")
    
    # Financial markets news
    usanews_query = f"financial markets stock market economy business news"
    print("\n💰 Financial Markets News:")
    news_update_usa = get_news_with_firecrawl(
        query=usanews_query,
        limit=10,
        time_filter="qdr:d",  # Past day
        country="US"
    )
    save_message_to_file(news_update_usa, "news_usa_report.txt")
    
    # India news
    india_news_query = f"India news latest politics economy"
    print("\n🇮🇳 India News:")
    news_update_india = get_news_with_firecrawl(
        query=india_news_query,
        limit=10,
        time_filter="qdr:d",  # Past day
        country="IN"  # India-specific results
    )
    save_message_to_file(news_update_india, "news_india_report.txt")
    
    # Voice format for news
    voicebot_updates = f"""
    Here are today's key updates across technology, financial markets, and India:
    
    Technology Updates:
    {news_update_tech}

    Financial Market Headlines:
    {news_update_usa}

    Latest from India:
    {news_update_india}
    
    Please present this information in a natural, conversational way suitable for speaking.
    """
    
    print("\n🎙️ Generating voice-friendly news...")
    news_voicebot_response = generate_gpt_response_voicebot(voicebot_updates, max_tokens=500)
    save_message_to_file(news_voicebot_response, "news_voicebot_report.txt")
    
    # Generate audio for news
    print("\n🔊 Generating audio for news updates...")
    news_audio_path = "news_update_report.wav"
    if text_to_speech_nvidia(news_voicebot_response, news_audio_path):
        print("✓ News audio generated")
    
    print("\n" + "="*60)
    print("✅ All reports generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  • bing_data/year_progress_report.txt")
    print("  • year_progress_report.wav")
    print("  • year_progress_report.mp3")
    print("  • bing_data/news_tech_report.txt")
    print("  • bing_data/news_usa_report.txt")
    print("  • bing_data/news_india_report.txt")
    print("  • bing_data/news_voicebot_report.txt")
    print("  • news_update_report.wav")
    print("  • news_update_report.mp3")
    print("="*60 + "\n")

