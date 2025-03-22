from astropy.modeling import custom_model

from sort import Sort
from ultralytics import YOLO
import torch
import cv2
import numpy as np


class Tracker:
    def __init__(self, path, device, threshold, custom_model_path, yolo_model_path):
        self.path = path
        self.device = device
        self.threshold = threshold

        self.custom_model = self.load_model(custom_model_path)
        self.yolo_model = self.load_model(yolo_model_path)

        self.custom_names = self.custom_model.names
        self.yolo_model = self.yolo_model.names


        self.tracker = Sort(max_age=1000, min_hits=3, iou_threshold=0.4)


    def load_model(self, model_path):
        model = YOLO(model_path)
        model.to(self.device)
        model.fuse()
        return model

    def results(self, model, frame):
        results = model(frame)[0]
        return results

    def get_results(self, results):
        res_array = []
        for res in results:
            bbox = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()

            if conf > self.threshold:
                array = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], conf[0], cls[0]]
                res_array.append(array)

        return np.array(res_array)


    def detect(self, res_array):

        if len(res_array) == 0:
            res_array = np.empty((0, 5))

        detections = []
        res = self.tracker.update(res_array)

        bboxes = res[:, :-1]
        idc = res[:, -1].astype(int)
        classes = res_array[:, -1].astype(int)

        for bbox, idx, cls in zip(bboxes, idc, classes):
            detections.append([bbox, idx, cls])

        return detections


    def draw(self, frame, detections, names):
        if len(detections) > 0 and detections is not None:
            # print(detections)
            for detection in detections:
                # print(detection)
                bbox, idx, class_id = detection
                x1, y1, x2, y2 = map(int, bbox)

                name = f"{names[int(class_id)]}"
                text = f"{int(idx)}{name}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return frame


        else:
            return frame


    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(self.custom_model, frame)
            detections = self.get_results(results)
            detected_objects = self.detect(detections)
            frame = self.draw(frame, detected_objects, self.custom_names)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


file = "/Users/maxkucher/opencv/howitzer_detector/videos/video_6.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"

custom_model = "/Users/maxkucher/opencv/howitzer_detector/sort/artillery_detecor.pt"
yolo_model = "/Users/maxkucher/opencv/howitzer_detector/sort/yolo11l.pt"
# device = "cpu"
tracker = Tracker(file, device, 0.4, custom_model, yolo_model)
tracker()








