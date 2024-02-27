import cv2
from ultralytics import YOLO
import supervision as sv

height = 720
width = 1200


box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)


def main():
    #initialize the web camera
    cap = cv2.VideoCapture(0)
    #set the frame for camera window
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    #load V8 model
    model = YOLO("yolov8s.pt")

    #to read frame by frame from camera
    while True:
        ret, frame = cap.read()
        #detection task
        result = model(frame, agnostic_nms=True)[0]
        detection = sv.Detections.from_yolov8(result)

        #extract the labels looping through every detection
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detection
        ]

        #annotate those labels on camera window(use supervision)
        frame = box_annotator.annotate(
            scene=frame,
            detections=detection,
            labels=labels
        )

        #show annotation on camera window
        cv2.imshow('yolov8', frame)

        #to breake the camera window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    main()