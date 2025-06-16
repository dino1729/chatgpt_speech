#!/usr/bin/env bash
#
# Script to check the status of the RPi VoiceBot.

# Change into bot directory
cd "$(dirname "$0")" || exit 1

PROCESS_NAME="simple_voicebot_rpi.py"
PID=$(pgrep -f "$PROCESS_NAME")

if [ -n "$PID" ]; then
  echo "VoiceBot ($PROCESS_NAME) is running with PID $PID."
else
  echo "VoiceBot ($PROCESS_NAME) is not running."
fi
