from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data=r"C:\scare_ai\animal_dataset",
        epochs=15,
        imgsz=224,
        batch=8,
        device="cpu",
        project=r"C:\scare_ai\animal_models",
        name="animal_classifier_v1"
    )

if __name__ == "__main__":
    main()