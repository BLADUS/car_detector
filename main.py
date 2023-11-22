import cv2
import inference
import supervision as sv
import yaml

# Загрузите файл data.yaml
with open("/home/osada/project_prog/Python/car_detector/data.yaml", "r") as file:
    config = yaml.safe_load(file)

# Извлеките API-ключ из конфигурации
API_KEY = config["roboflow"]["api_key"]

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    cv2.imshow(
        "Prediction", 
        annotator.annotate(
            scene=image, 
            detections=detections,
            labels=labels
        )
    ),
    cv2.waitKey(1)

# Подставьте API-ключ в качестве аргумента при инициализации объекта Stream
inference.Stream(
    source="/home/osada/project_prog/Python/car_detector/video/crossroad4.mp4",
    model="vehiclesviewfromabove/3",
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=on_prediction,
    api_key=API_KEY
)
