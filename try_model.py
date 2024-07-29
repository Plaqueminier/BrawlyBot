import cv2
import numpy as np
from ultralytics import YOLO
import pyautogui
import time

SCREEN_REGION = (200, 140, 1200, 675)


def capture_screenshot():
    # Capture the entire screen
    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    # Convert the screenshot to a numpy array
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def draw_boxes(image, results):
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            label = f"{result.names[cls]} {conf:.2f}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    return image


def main():
    # Load your trained model
    model = YOLO("/Users/plqmnr/Documents/BrawlyBot/best.pt")  # Update this path

    while True:
        # Capture screenshot
        image = capture_screenshot()

        # Run inference
        results = model(image, conf=0.1)

        # Draw bounding boxes
        annotated_image = draw_boxes(image, results)

        # Display the result
        cv2.imshow("Brawl Stars Detection", annotated_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Optional: add a small delay to reduce CPU usage
        time.sleep(0.1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
