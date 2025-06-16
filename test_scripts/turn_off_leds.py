try:
    from aiy.leds import Leds, Pattern, RgbLeds, Color
    LEDS_AVAILABLE = True
except ImportError:
    print("Failed to import aiy.leds. LED functionality will be disabled.")
    LEDS_AVAILABLE = False
    # Define dummy Color class for compatibility if aiy.leds is not found
    class Color:
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        YELLOW = (255, 255, 0)
        CYAN = (0, 255, 255)
        MAGENTA = (255, 0, 255)

def force_led_off():
    if not LEDS_AVAILABLE:
        print("aiy.leds library not available. Cannot control LEDs.")
        return

    print("Attempting to turn off LEDs using aiy.leds...")
    try:
        with Leds() as leds:
            leds.update(Leds.rgb_off())
        print("LEDs should now be off.")
    except Exception as e:
        print(f"Error trying to turn off LEDs with aiy.leds: {e}")
        print("You might also need to ensure no other script is controlling the LEDs.")
        print("As a last resort, a full reboot of the Raspberry Pi usually resets LED states.")

if __name__ == '__main__':
    force_led_off()

    # As an additional measure, also try GPIO cleanup
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False) # Suppress warnings
        GPIO.cleanup()
        print("RPi.GPIO cleanup also attempted.")
    except ImportError:
        print("RPi.GPIO library not found, skipping GPIO cleanup.")
    except Exception as e:
        print(f"Error during RPi.GPIO cleanup: {e}")
