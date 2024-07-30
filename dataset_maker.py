import mss
from PIL import Image
import time
import os

# Screenshot area (adjust as needed)
SCREEN_AREA = {"top": 80, "left": 0, "width": 1200, "height": 675}

# Ensure the screenshot directory exists
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Initialize mss for screenshots
sct = mss.mss()


def capture_screenshots():
    screenshot_interval = 5  # seconds
    try:
        while True:
            # Capture the specified region
            screenshot = sct.grab(SCREEN_AREA)
            # Convert to PIL Image
            img = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )
            # Save the screenshot
            filename = f"screenshots/screenshot_{int(time.time())}.png"
            img.save(filename)
            print(f"Screenshot saved: {filename}")

            # Wait for the next screenshot interval
            time.sleep(screenshot_interval)
    except KeyboardInterrupt:
        print("Screenshot capture stopped.")


if __name__ == "__main__":
    print("Starting screenshot capture. Press Ctrl+C to stop.")
    capture_screenshots()
