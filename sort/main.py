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


    def detection_area(self, frame, detections):
        if detections is not None:
            if len(detections) == 1:
                for detection in detections:
                    bbox, idx, class_id = detection
                    x1, y1, x2, y2 = map(int, bbox)

                    new_x1 = x1 - 40
                    new_y1 = y1 - 40
                    new_x2 = x2 + 40
                    new_y2 = y2 + 40

                    upd_frame = frame[new_y1:new_y2, new_x1:new_x2]
                    cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)

            elif len(detections) > 1:

                x1s, y1s, x2s, y2s = [], [], [], []
                for detection in detections:

                    bbox, idx, class_id = detection
                    x1, y1, x2, y2 = map(int, bbox)

                    x1s.append(x1)
                    y1s.append(y1)
                    x2s.append(x2)
                    y2s.append(y2)

                min_x1 = min(x1s)-40
                min_y1 = min(y1s)-40
                max_x2 = max(x2s)+40
                max_y2 = max(y2s)+40

                upd_frame = frame[min_y1:max_y2, min_x1:max_x2]
                cv2.rectangle(frame, (min_x1, min_y1), (max_x2, max_y2), (0, 255, 0), 2)





                ...
        return None


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
            self.detection_area(frame, detected_objects)

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








