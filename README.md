# 🤖 Orion Voice Assistant
### *Your Personal AI Companion on Raspberry Pi Zero 2W*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Zero%202W-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Transform your Raspberry Pi Zero 2W into an intelligent voice assistant with the power of OpenAI's latest GPT-4o-mini model!**

[🎥 Watch Demo Video](#demo) • [⚡ Quick Start](#quick-start) • [🛠️ Hardware Setup](#hardware-requirements) • [📖 Documentation](#documentation)

</div>

---

## 🌟 What Makes Orion Special?

Orion isn't just another voice assistant - it's a **cutting-edge AI companion** that brings the full power of OpenAI's GPT-4o-mini directly to your Raspberry Pi Zero 2W. With **native audio processing**, **intelligent LED feedback**, and **seamless voice interactions**, Orion represents the perfect fusion of compact hardware and advanced AI.

### ✨ Key Features

🎙️ **Natural Voice Interactions** - Talk naturally, get intelligent responses  
🔊 **High-Quality Audio Processing** - Crystal clear voice recognition and synthesis  
💡 **Smart LED Feedback** - Visual indicators for different states (listening, processing, responding)  
🎯 **Button-Triggered Recording** - Hardware button for precise voice capture control  
🧠 **GPT-4o-mini Integration** - Latest OpenAI model with audio capabilities  
⚡ **Ultra-Lightweight** - Optimized for Raspberry Pi Zero 2W performance  
🎵 **Multiple Voice Options** - 8 different AI voices to choose from  
🔄 **Auto-Reset Capability** - Smart conversation management based on token count  
📱 **Headless Operation** - Perfect for embedded applications  

---

## 🎥 Demo

### 📱 Watch Orion in Action!

https://github.com/user-attachments/assets/02a2a1a8-d76b-4aa3-9b25-fffffbeb9d71

*Orion Voice Assistant running on Raspberry Pi Zero 2W - demonstrating natural voice interactions, LED feedback, and GPT-4o-mini responses in real-time!*

> 🎬 **Demo highlights:** Voice activation, LED status indicators, natural conversation flow, and lightning-fast AI responses on compact hardware.

---

## ⚡ Quick Start

Get Orion running on your Raspberry Pi Zero 2W in just 5 minutes!

### 1. Clone & Setup
```bash
git clone https://github.com/dino1729/chatgpt_speech.git
cd chatgpt_speech
```

### 2. Install Dependencies (Pi Zero Optimized)
```bash
pip install -r requirements_pizero.txt
```

### 3. Configure API Keys
```bash
cp config/config.example.yml config/config.yml
# Edit config.yml with your OpenAI API key
```

### 4. Launch Orion
```bash
python simple_voicebot_rpi.py
```

**That's it!** Your AI companion is ready to talk! 🎉

---

## 🛠️ Hardware Requirements

### Minimum Setup (Raspberry Pi Zero 2W)
- **Raspberry Pi Zero 2W** (recommended) or Pi 3B+
- **MicroSD Card** (32GB+ recommended)
- **USB Microphone** or **Google AIY Voice HAT**
- **Speaker** or **3.5mm Audio Output**
- **Push Button** (connected to GPIO pin 23)
- **LED Strip** (optional, for visual feedback)

### Recommended: Google AIY Voice Kit
For the best experience, we recommend using the **Google AIY Voice Kit** which provides:
- High-quality microphone array
- Built-in speaker
- RGB LED ring for visual feedback
- Pre-configured button setup

---

## 🏗️ Project Structure

```
chatgpt_speech/
├── simple_voicebot_rpi.py    # 🎯 Main Pi Zero 2W script
├── simple_voicebot.py        # Core voice bot logic
├── requirements_pizero.txt   # Minimal dependencies for Pi Zero
├── requirements.txt          # Full feature requirements
├── config/
│   ├── config.yml           # API keys and settings
│   ├── prompts.yml          # AI personality configuration
│   └── config.py            # Configuration loader
├── start_voicebot_rpi.sh    # 🚀 Auto-start script
├── stop_voicebot_rpi.sh     # ⏹️ Stop script
└── status_voicebot_rpi.sh   # 📊 Status checker
```

---

## 🎛️ Configuration Options

### Voice Personalities
Customize Orion's personality by editing `config/prompts.yml`:
- **Default**: Knowledgeable assistant
- **Friendly**: Casual conversation partner
- **Professional**: Business-focused responses
- **Creative**: Artistic and imaginative

### Voice Selection
Choose from 8 different AI voices:
`alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`

### LED Feedback Colors
- 🟢 **Green**: Ready/Idle
- 🔵 **Blue**: Listening
- 🟡 **Yellow**: Processing
- 🔴 **Red**: Error/Offline

---

## 🔧 Advanced Setup

### Auto-Start on Boot
```bash
# Add to crontab for auto-start
@reboot /home/pi/chatgpt_speech/start_voicebot_rpi.sh
```

### Service Management
```bash
# Start Orion
./start_voicebot_rpi.sh

# Check status
./status_voicebot_rpi.sh

# Stop Orion
./stop_voicebot_rpi.sh
```

### Audio Device Configuration
```bash
# List available audio devices
aplay -l
arecord -l

# Test audio setup
aplay /usr/share/sounds/alsa/Front_Left.wav
arecord -d 5 test.wav && aplay test.wav
```

---

## 📋 Requirements

### Software Dependencies
- **Python 3.9+**
- **OpenAI Python SDK** (>=1.40.0)
- **RPi.GPIO** (for button control)
- **PyYAML** (configuration management)

### API Requirements
- **OpenAI API Key** (GPT-4o-mini access)
- Internet connection for API calls

---

## 🎯 Use Cases

### Home Automation
- Voice control for smart devices
- Information queries and weather updates
- Daily briefings and reminders

### Educational Projects
- Interactive learning companion
- STEM project demonstrations
- AI and robotics education

### Accessibility
- Voice interface for mobility-impaired users
- Audio feedback for visually impaired users
- Hands-free device control

### IoT Applications
- Smart mirror integration
- Voice-controlled robots
- Embedded AI solutions

---

## 🐛 Troubleshooting

### Common Issues

**Audio not working?**
```bash
# Check audio devices
sudo cat /proc/asound/cards
alsamixer  # Adjust volume levels
```

**Button not responding?**
```bash
# Check GPIO pin configuration
gpio readall  # Verify pin 23 setup
```

**API errors?**
- Verify OpenAI API key in `config/config.yml`
- Check internet connection
- Ensure sufficient API credits

**Memory issues on Pi Zero?**
```bash
# Increase swap file size
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## 🤝 Contributing

We love contributions! Here's how you can help make Orion even better:

1. **🍴 Fork** the repository
2. **🌟 Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **✅ Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **📤 Push** to the branch (`git push origin feature/AmazingFeature`)
5. **🔀 Open** a Pull Request

### Ideas for Contributions
- 🔌 Additional hardware integrations
- 🎨 New voice personalities
- 🌍 Multi-language support
- 📱 Mobile app companion
- 🏠 Home Assistant integration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for the incredible GPT-4o-mini model
- **Google** for the AIY Voice Kit hardware platform
- **Raspberry Pi Foundation** for making AI accessible to everyone
- **Open Source Community** for inspiration and support

---

<div align="center">

**Ready to build your own AI companion?**

⭐ **Star this repo** if you found it helpful!  
🐛 **Report issues** to help us improve  
💬 **Join the discussion** in our community  

[Get Started](#quick-start) • [View Demo](#demo) • [Join Community](https://github.com/dino1729/chatgpt_speech/discussions)

---

*Built with ❤️ by the Open Source Community*

</div>
