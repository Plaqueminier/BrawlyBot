from ultralytics import YOLO
import yaml
import os


def create_dataset_yaml(dataset_path, class_names):
    yaml_content = {
        "train": os.path.join(dataset_path, "train", "images"),
        "val": os.path.join(dataset_path, "valid", "images"),
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    return yaml_path


def train_brawl_stars_model():
    # Define your dataset path and class names
    dataset_path = "/Users/plqmnr/Documents/BrawlyBot/dataset/"
    class_names = [
        "Me",
        "Enemy",
        "Bush",
        "Wall",
        "Water",
        "Box",
        "Power cube",
        "Gas",
        "Bear ally",
        "Bear enemy",
        "Turret ally",
        "Turret enemy",
    ]  # Update with your class names

    # Create dataset.yaml file
    yaml_path = create_dataset_yaml(dataset_path, class_names)

    # Load a pre-trained YOLOv8 model
    model = YOLO(
        "yolov8n.pt"
    )  # 'n' for nano, can be 's' for small, 'm' for medium, etc.

    # Train the model
    results = model.train(
        data=yaml_path, epochs=50, imgsz=600, batch=16, name="brawl_stars_model", device="mps"
    )

    results = model.val()

    model.export(format="torchscript")

    print(results)


if __name__ == "__main__":
    train_brawl_stars_model()
