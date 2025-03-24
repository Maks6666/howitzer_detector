
from sort import Sort
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from analyzer.analyzer_model import model


class Tracker:
    def __init__(self, path, device, threshold, detection_rectangle, custom_model_path, yolo_model_path):
        self.path = path
        self.device = device
        self.threshold = threshold

        self.custom_model = self.load_model(custom_model_path)
        self.yolo_model = self.load_model(yolo_model_path)

        self.custom_names = self.custom_model.names
        self.yolo_names = self.yolo_model.names


        self.custom_tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.4)
        self.yolo_tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.4)

        self.feature_list = ["person", "car", "bus", "truck"]
        self.status = ["decoy", "unsure", "hofitzer"]

        self.detection_rectangle = detection_rectangle


    def load_model(self, model_path):
        model = YOLO(model_path)
        model.to(self.device)
        model.fuse()
        return model

    def results(self, model, frame):

        results = model(frame, classes=[0, 2, 5, 7])[0]

        return results

    def get_results(self, results):
        if results is not None and len(results) > 0:
            res_array = []
            for res in results:
                bbox = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()

                if conf > self.threshold:
                    array = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], conf[0], cls[0]]
                    res_array.append(array)

            return np.array(res_array)
        return None


    def detect(self, tracker, res_array):
        if res_array is not None and len(res_array) > 0:
            if len(res_array) == 0:
                res_array = np.empty((0, 5))

            detections = []
            res = tracker.update(res_array)

            bboxes = res[:, :-1]
            idc = res[:, -1].astype(int)
            classes = res_array[:, -1].astype(int)

            for bbox, idx, cls in zip(bboxes, idc, classes):
                detections.append([bbox, idx, cls])

            return detections

        else:
            return None


    def draw(self, frame, detections):
        if detections is not None:
            if len(detections) > 0:
                for detection in detections:
                    bbox, idx, class_id = detection
                    x1, y1, x2, y2 = map(int, bbox)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                return frame

            else:
                return frame
        else:
            return frame




    def detection_area(self, frame, detections):
        if detections is not None:
            if len(detections) == 1:
                for detection in detections:
                    bbox, idx, class_id = detection
                    x1, y1, x2, y2 = map(int, bbox)

                    new_x1 = max(x1 - self.detection_rectangle, 0)
                    new_y1 = max(y1 - self.detection_rectangle, 0)
                    new_x2 = min(x2 + self.detection_rectangle, frame.shape[1])
                    new_y2 = min(y2 + self.detection_rectangle, frame.shape[0])

                    upd_frame = frame[new_y1:new_y2, new_x1:new_x2]
                    # cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)

                    return upd_frame

            elif len(detections) > 1:

                x1s, y1s, x2s, y2s = [], [], [], []
                for detection in detections:

                    bbox, idx, class_id = detection
                    x1, y1, x2, y2 = map(int, bbox)

                    x1s.append(x1)
                    y1s.append(y1)
                    x2s.append(x2)
                    y2s.append(y2)

                min_x1 = max(min(x1s) - self.detection_rectangle, 0)
                min_y1 = max(min(y1s) - self.detection_rectangle, 0)
                max_x2 = min(max(x2s) + self.detection_rectangle, frame.shape[1])
                max_y2 = min(max(y2s) + self.detection_rectangle, frame.shape[0])

                upd_frame = frame[min_y1:max_y2, min_x1:max_x2]
                # cv2.rectangle(frame, (min_x1, min_y1), (max_x2, max_y2), (0, 255, 0), 2)

                return upd_frame


            elif len(detections) == 0:
                return frame

        return frame

    def count_objects(self, frame, yolo_detections, custom_detections):

        targets = {}

        objects = {
            "personal": [],
            "cars": [],
            "trucks": [],
            "busses": []
        }

        if frame is not None:
            if yolo_detections is not None and len(yolo_detections) > 0:
                for detection in yolo_detections:
                    _, _, class_id = detection
                    name = self.yolo_names[int(class_id)]

                    if name in self.feature_list:

                        if name == "person":
                            objects["personal"].append(name)

                        if name == "car":
                            objects["cars"].append(name)

                        if name == "truck":
                            objects["trucks"].append(name)

                        if name == "bus":
                            objects["busses"].append(name)


                        # ---------------------------------------------

                personal = 1 if len(objects["personal"]) > 0 else 0
                cars = 1 if len(objects["cars"]) > 0 else 0
                trucks = 1 if len(objects["trucks"]) > 0 else 0
                busses = 1 if len(objects["busses"]) > 0 else 0

                array = np.array([personal, cars, trucks, busses])
                array = array.reshape(1, -1)
                res = model.predict(array)
                pred = self.status[int(res)]


                for detection in custom_detections:
                    _, idx, class_id = detection
                    targets[idx] = pred

                return objects, targets




            elif yolo_detections is not None and len(yolo_detections) == 0:
                personal = 0
                cars = 0
                trucks = 0
                busses = 0

                array = np.array([personal, cars, trucks, busses])
                array = array.reshape(1, -1)
                res = model.predict(array)
                pred = self.status[int(res)]

                for detection in custom_detections:
                    _, idx, class_id = detection
                    targets[idx] = pred


                return objects, targets


            else:
                return objects, targets


        else:
            return objects, targets


    def assign_status(self, frame, targets, detections):
        status = ""
        color = (0, 0, 0)
        if detections is not None:
            for detection in detections:
                bbox, idx, class_id = detection
                x1, y1, x2, y2 = map(int, bbox)



                for key, value in targets.items():
                    if key == idx:
                        status = value
                        if status == "decoy":
                            color = (0, 255, 0)
                        elif status == "unsure":
                            color = (255, 255, 0)
                        elif status == "hofitzer":
                            color = (0, 0, 255)

                text = f"{idx}:object:{status}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)




    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(self.custom_model, frame)
            detections = self.get_results(results)
            detected_objects = self.detect(self.custom_tracker, detections)
            frame = self.draw(frame, detected_objects)


            upd_frame = self.detection_area(frame, detected_objects)
            detected_yolo_objects = None



            if upd_frame is not None and upd_frame.size > 1:
                yolo_results = self.results(self.yolo_model, upd_frame)
                yolo_detections = self.get_results(yolo_results)
                detected_yolo_objects = self.detect(self.yolo_tracker, yolo_detections)


            objects, targets = self.count_objects(frame, detected_yolo_objects, detected_objects)
            _ = self.draw(upd_frame, detected_yolo_objects)
            self.assign_status(frame, targets, detected_objects)
            # print(objects, targets)


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
tracker = Tracker(file, device, 0.4, 50, custom_model, yolo_model)
tracker()







