# Raspberry Pi Voice Assistant with GPT-3 Integration

This repository contains a Python script that sets up a voice-controlled chatbot using OpenAI's GPT-3 model. This was built with Google AIY Voice Kit and Raspberry Pi 3 Model B+.
Please refer to the AIY Voice Kit page for documentation on how to set up the hardware.

## Features

- Voice-controlled interaction with the chatbot using a physical button.
- Automatic reset of the conversation to the default prompt based on token count or inactivity.
- Audio recording and transcription for user input.
- Translation of bot responses to the detected language of the user input.
- Integration with GPT-3.5 Turbo for generating assistant replies.
- Text-to-speech conversion of bot responses for verbal interaction.

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone https://github.com/dino1729/chatgpt_speech.git
cd chatgpt_speech
```
2. Setup dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Set up the necessary API keys and run the script:

```bash
python voicebot.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
