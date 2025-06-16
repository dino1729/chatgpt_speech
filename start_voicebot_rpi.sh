#!/usr/bin/env bash
#
# Simple wrapper to launch the RPi VoiceBot with nohup and discard logs.

# Change into bot directory
cd "$(dirname "$0")" || exit 1

# Prevent multiple copies
if pgrep -f simple_voicebot_rpi.py >/dev/null; then
  echo "VoiceBot is already running."
  exit 0
fi

# Launch unbuffered, backgrounded, with no output
nohup /home/pi/python39/bin/python3.9 -u simple_voicebot_rpi.py \
  > /dev/null 2>&1 &

echo "VoiceBot started (PID $!)."
