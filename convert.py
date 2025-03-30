from ultralytics import YOLO


def convert_model(path: str):
    model = YOLO(path)
    model.export(format="onnx")


if __name__ == "__main__":
    convert_model("yolo11n.pt")
