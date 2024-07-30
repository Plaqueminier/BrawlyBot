import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO


def load_model(weights_path):
    """Load the YOLO model."""
    return YOLO(weights_path)


def process_image(model, image_path, conf_threshold=0.25, ignore_classes=None):
    """Process a single image and return detections."""
    img = Image.open(image_path)
    results = model(img, conf=conf_threshold)
    detections = results[0].boxes.data  # returns x1, y1, x2, y2, conf, class

    if ignore_classes:
        mask = torch.ones(detections.shape[0], dtype=torch.bool)
        for cls in ignore_classes:
            mask &= detections[:, 5] != cls
        detections = detections[mask]

    return detections


def convert_to_yolo_format(detections, img_width, img_height):
    """Convert detections to YOLO format."""
    yolo_annotations = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        # Convert to center format
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        yolo_annotations.append(f"{int(cls)} {x_center} {y_center} {width} {height}")
    return yolo_annotations


def draw_boxes(image, detections, class_names):
    """Draw bounding boxes on the image."""
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = (0, 255, 0)  # Green color for the box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[int(cls)]} {conf:.2f}"
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return image


def process_directory(
    model,
    input_dir,
    output_dir,
    visualized_dir,
    conf_threshold=0.25,
    ignore_classes=None,
):
    """Process all images in a directory and save annotations and visualized images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(visualized_dir):
        os.makedirs(visualized_dir)

    class_names = model.names  # Get class names from the model

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)
            height, width = img.shape[:2]

            detections = process_image(
                model, image_path, conf_threshold, ignore_classes
            )
            yolo_annotations = convert_to_yolo_format(detections, width, height)

            # Save annotations
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_annotations))

            # Save visualized image
            visualized_img = draw_boxes(img.copy(), detections, class_names)
            vis_path = os.path.join(visualized_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(vis_path, visualized_img)

            print(f"Processed {filename}")


if __name__ == "__main__":
    model_path = (
        "/Users/plqmnr/Documents/BrawlyBot/best.pt"  # Update this with your model path
    )
    input_directory = "/Users/plqmnr/Documents/BrawlyBot/val1"  # Update this with your input images directory
    output_directory = "/Users/plqmnr/Documents/BrawlyBot/yolo/obj_Validation_data"  # Update this with your desired output directory
    visualized_directory = "/Users/plqmnr/Documents/BrawlyBot/visualized"  # Update this with your desired output directory for visualized images
    confidence_threshold = 0.25  # Adjust this threshold as needed
    ignore_classes = []  # Add the class IDs you want to ignore

    model = load_model(model_path)
    process_directory(
        model,
        input_directory,
        output_directory,
        visualized_directory,
        confidence_threshold,
        ignore_classes,
    )
    print("Pre-annotation complete!")
