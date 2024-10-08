import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import AzureOpenAI as OpenAIAzure
from pyowm import OWM
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.audio_processors import text_to_speech_nospeak
from datetime import datetime
import random

azure_api_key = config.azure_api_key
azure_api_base = config.azure_api_base
azure_chatapi_version = config.azure_chatapi_version
azure_chatapi_version = config.azure_chatapi_version
# azure_gpt4omini_deploymentid = config.azure_gpt4omini_deploymentid
azure_gpt4_deploymentid = config.azure_gpt4_deploymentid

yahoo_id = config.yahoo_id
yahoo_app_password = config.yahoo_app_password
pyowm_api_key = config.pyowm_api_key

temperature = config.temperature
max_tokens = config.max_tokens

model_names = ["BING+OPENAI", "GPT4OMINI", "GPT4", "GEMINI", "COHERE", "MIXTRAL8x7B"]

def generate_gpt_response(user_message):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    syspromptmessage = f"""
    You are EDITH, or "Even Dead, I'm The Hero," a world-class AI assistant that is designed by Tony Stark to be a powerful tool for whoever controls it. You help Dinesh in various tasks. Your response will be converted into speech and will be played on Dinesh's smart speaker. Your responses must reflect Tony's characteristic mix of confidence and humor. Start your responses with a unique, witty and engaging introduction to grab the Dinesh's attention.
    """
    system_prompt = [{
        "role": "system",
        "content": syspromptmessage
    }]
    conversation = system_prompt.copy()
    conversation.append({"role": "user", "content": str(user_message)})
    response = client.chat.completions.create(
        model=azure_gpt4_deploymentid,
        messages=conversation,
        max_tokens=1024,
        temperature=0.3,
    )
    message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": str(message)})

    return message

def generate_quote():
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    quote_prompt = """
    Provide a random quote from either Chanakya, Winston Churchill, Lord Krishna, Richard Feynman, Albert Einstein, Nikola Tesla, Marie Curie, Alan Turing, Carl Sagan, Leonardo da Vinci, Douglas Engelbart, JCR Licklider, or Vannevar Bush.
    """
    response = client.chat.completions.create(
        model=azure_gpt4_deploymentid,
        messages=[{"role": "user", "content": quote_prompt}],
        max_tokens=60,
        temperature=0.5,
    )
    quote = response.choices[0].message.content.strip()
    return quote

def get_weather():
    owm = OWM(pyowm_api_key)
    mgr = owm.weather_manager()
    weather = mgr.weather_at_id(5743413).weather  # North Plains, OR
    temp = weather.temperature('celsius')['temp']
    status = weather.detailed_status
    return temp, status

def generate_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left):
    # Weather setup
    temp, status = get_weather()

    # Date and time
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")

    # Quarterly earnings dates (based on MSFT's reporting)
    earnings_dates = [
        datetime(2024, 1, 23),  # Q1 end, Q2 start
        datetime(2024, 4, 25),  # Q2 end, Q3 start
        datetime(2024, 7, 29),  # Q3 end, Q4 start
        datetime(2024, 10, 24), # Q4 end, Q1 start (assumed)
        datetime(2025, 1, 23)   # Next Q1 (assumed)
    ]

    # Determine the current quarter based on the date
    current_quarter = None
    for i in range(len(earnings_dates) - 1):
        if earnings_dates[i] <= now < earnings_dates[i + 1]:
            current_quarter = i + 1
            start_of_quarter = earnings_dates[i]
            end_of_quarter = earnings_dates[i + 1]
            break

    # Days and progress in the current quarter
    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    percent_days_left_in_quarter = ((days_in_quarter - days_completed_in_quarter) / days_in_quarter) * 100

    # Progress bar visualization
    progress_bar_full = '█'
    progress_bar_empty = '░'
    progress_bar_length = 20
    quarter_progress_filled_length = int(progress_bar_length * (100 - percent_days_left_in_quarter) / 100)
    quarter_progress_bar = progress_bar_full * quarter_progress_filled_length + progress_bar_empty * (progress_bar_length - quarter_progress_filled_length)

    progress_filled_length = int(progress_bar_length * (100 - percent_days_left) / 100)
    progress_bar = progress_bar_full * progress_filled_length + progress_bar_empty * (progress_bar_length - progress_filled_length)

    return f"""

    Year Progress Report

    Today's Date and Time: {date_time}
    Weather in North Plains, OR: {temp}°C, {status}

    Current Quarter: Q{current_quarter}

    Q{current_quarter} Progress: [{quarter_progress_bar}] {100 - percent_days_left_in_quarter:.2f}% completed

    Days completed in the year: {days_completed}
    Weeks completed in the year: {weeks_completed:.2f}

    Days left in the year: {days_left}
    Weeks left in the year: {weeks_left:.2f}
    Percentage of the year left: {percent_days_left:.2f}%

    Year Progress: [{progress_bar}] {100 - percent_days_left:.2f}% completed

    """

def send_email(subject, message):
    sender_email = yahoo_id
    receiver_email = "katam.dinesh@hotmail.com"
    password = yahoo_app_password

    email_message = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject

    email_message.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP('smtp.mail.yahoo.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = email_message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"Email sent successfully with subject: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def time_left_in_year():
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

if __name__ == "__main__":

    days_completed, weeks_completed, days_left, weeks_left, percent_days_left = time_left_in_year()
    year_progress_message = generate_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left)
    # print(year_progress_message)

    quote = generate_quote()
    year_progress_message_with_quote = f"{year_progress_message}\n\nQuote of the Day: {quote}"
    
    year_progress_subject = "Year Progress Report 📅"
    send_email(year_progress_subject, year_progress_message_with_quote)

    year_progress_message_prompt = f"""
    Here is a year progress report for {datetime.now().strftime("%B %d, %Y")}:

    {year_progress_message}

    Please analyze the report and summarize it in a manner suitable for a voice assistant to deliver as a daily update. At the end of the summary, include a random quote from either Chanakya, Winston Churchill, Lord Krishna, Richard Feynman, Albert Einstein, Nikola Tesla, Marie Curie, Alan Turing, Carl Sagan, Leonardo da Vinci, Douglas Engelbart, JCR Licklider, or Vannevar Bush.
    """

    year_progress_gpt_response = generate_gpt_response(year_progress_message_prompt)
    # print(f"\nGPT Response:\n {year_progress_gpt_response}")

    # Convert the year progress report to speech
    yearprogress_tts_output_path = "year_progress_report.mp3"
    model_name = random.choice(model_names)
    text_to_speech_nospeak(year_progress_gpt_response, yearprogress_tts_output_path, model_name=model_name)

    # News Updates
    news_update_subject = "News Updates 📰"
    technews = "Latest news in technology"
    news_update_tech = internet_connected_chatbot(technews, [], model_name, max_tokens, temperature)
    usanews = "Latest news in Financial Markets"
    news_update_usa = internet_connected_chatbot(usanews, [], model_name, max_tokens, temperature)
    india_news = "Latest news from India"
    news_update_india = internet_connected_chatbot(india_news, [], model_name, max_tokens, temperature)

    # Collate all news updates and send them in an email after processing them with gpt
    news_updates = f"""
    Here are the latest news updates in various categories for {datetime.now().strftime("%B %d, %Y")}:

    Tech News Update:
    {news_update_tech}

    Financial Markets News Update:
    {news_update_usa}

    India News Update:
    {news_update_india}

    Analyze the news updates and provide a brief 5 key-point summary for each news category. Keep the summary very concise and include only the headline news.
    """

    # print(f"\nDetailed News Updates:\n {news_updates}")

    news_gpt_response = generate_gpt_response(news_updates)
    # print(f"\nSummarized News Update:\n {news_gpt_response}")
    
    send_email(news_update_subject, news_gpt_response)

    # Convert the news updates to speech
    news_tts_output_path = "news_update_report.mp3"
    model_name = random.choice(model_names)
    text_to_speech_nospeak(news_gpt_response, news_tts_output_path, model_name=model_name)
