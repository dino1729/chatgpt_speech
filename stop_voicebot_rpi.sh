#!/usr/bin/env bash
#
# Script to stop the RPi VoiceBot.

# Change into bot directory
cd "$(dirname "$0")" || exit 1

PROCESS_NAME="simple_voicebot_rpi.py"
PID=$(pgrep -f "$PROCESS_NAME")

if [ -n "$PID" ]; then
  echo "Stopping VoiceBot ($PROCESS_NAME) with PID $PID..."
  kill "$PID"
  # Optionally, add a loop to wait for the process to terminate
  # and then use kill -9 "$PID" if it's still running.
  echo "VoiceBot stopped."
else
  echo "VoiceBot ($PROCESS_NAME) is not running."
fi

# Attempt to turn off LEDs
echo "Attempting to turn off LEDs..."
if python3 test_scripts/turn_off_leds.py; then
  echo "LED turn off script executed."
else
  echo "Failed to execute LED turn off script or script reported an error."
fi
