import cv2
from panel.widgets.indicators import ptqdm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import random



class DeepDetector:
    def __init__(self, path, device, threshold, model_path, yolo_model_path):
        self.path = path
        self.device = device
        self.threshold = threshold

        self.model = self.load_model(model_path)
        self.yolo_model = self.load_model(yolo_model_path)

        self.names = self.model.names
        self.yolo_names = self.yolo_model.names


        self.custom_tracker = DeepSort(max_iou_distance=0.8, max_age=60, n_init=5)

        self.feature_list = ["person", "car", "bus", "truck", "target"]

    def load_model(self, model_path):
        model = YOLO(model_path)
        model.to(self.device)
        model.fuse()
        return model


    def results(self, model, frame):
        return model(frame)[0]

    def get_results(self, results, tracker, frame):
        res_array = []
        if len(results) !=0 and results is not None:
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > self.threshold:
                    res_array.append(([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)], float(score), int(class_id)))

            print(f"Results: {res_array}")
            tracks = tracker.update_tracks(raw_detections=res_array, frame=frame)
            # print(tracks)
            detected_objects = []
            for track in tracks:
                bbox = track.to_ltrb()
                idx = track.track_id
                class_id = track.get_det_class()
                detected_objects.append((bbox, idx, class_id))

            print(detected_objects)
            return detected_objects
        else:
            return None


    def count_objects(self, detected_objects, names):

        targets = {
            "decoys": [],
            "unsure": [],
            "targets": []
        }

        objects_counter = {
            "person": [],
            "truck": [],
            "car": [],
            "bus": []
        }

        if detected_objects is not None and len(detected_objects) > 0:
            # print(detected_objects)
            for _, idx, class_id in detected_objects:
                name = ""
                if class_id in names.keys():
                    name = names[int(class_id)]

                if name == "person":
                    objects_counter["person"].append(idx)

                if name == "truck":
                    objects_counter["truck"].append(idx)

                if name == "car":
                    objects_counter["car"].append(idx)

                if name == "bus":
                    objects_counter["bus"].append(idx)

                # --------------------------------------------------------------------------------------------------------------------------------------------

                if name == "target":
                    person_amount = len(objects_counter["person"])
                    # array = np.array([person_amount, ...])
                    # res = model.predict(array)
                    res = random.randint(0, 3)

                    if res == 0:
                        targets["decoys"].append(idx)

                    if res == 1:
                        targets["unsure"].append(idx)

                    if res == 2:
                        targets["targets"].append(idx)


            return objects_counter, targets

        else:
            return objects_counter, targets


    def draw(self, detected_objects, frame, names, targets):
        bboxes = []
        if detected_objects is not None and len(detected_objects) > 0:

            for bbox, idx, class_id in detected_objects:
                text = ""

                # bboxes.append(bbox)
                x1, y1, x2, y2 = map(int, bbox)
                # bbox = (x1, y1, x2, y2)
                сolor = (0, 255, 0)

                if class_id in names.keys():
                    name = names[int(class_id)]
                    if name in self.feature_list:
                        if name == "target":
                            сolor = (0, 0, 255)
                            for key, value in targets.items():
                                if idx in targets[key]:
                                    status = key
                                    text = f"{idx}:{name}:{status}"
                        bboxes.append(bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), сolor, 2)
                        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, сolor, 2)
                    # else:
                    #     text = f"{name}"

                    # bboxes.append(bbox)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), сolor, 2)
                    # cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, сolor, 2)

            # print(bbox)
            return frame, bboxes
        else:
            return frame, None

    def detect_objects(self, frame, bboxes):
        if bboxes is not None and len(bboxes) > 0:
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)

                new_x1 = max(0, x1 - (x2 - x1) // 0.5)
                new_y1 = max(0, y1 - (y2 - y1) // 0.5)
                new_x2 = min(frame.shape[1], x2 + (x2 - x1) // 0.5)
                new_y2 = min(frame.shape[0], y2 + (y2 - y1) // 0.5)
                # print(new_x1, new_x2, new_y1, new_y2)
                # cv2.rectangle(frame, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (0, 0, 255), 2)
                frame = frame[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]

            return frame
        else:
            return frame

    def display_objects(self, counter, frame):
        person_amount = len(counter["person"])
        car_amount = len(counter["car"])
        bus_amount = len(counter["bus"])
        truck_amount = len(counter["truck"])

        cv2.rectangle(frame, (0, 0), (90, 40), (0, 0, 0), 80)

        cv2.putText(frame, str(person_amount), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, str(car_amount), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, str(bus_amount), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, str(truck_amount), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(self.model, frame)

            detected_objects = self.get_results(results, self.custom_tracker, frame)
            counter, targets = self.count_objects(detected_objects, self.names)

            frame, bboxes = self.draw(detected_objects, frame, self.names, targets)

            if bboxes is not None:
                upd_frame = self.detect_objects(frame, bboxes)

                if upd_frame is not None and upd_frame.size > 0:

                    yolo_results = self.results(self.yolo_model, upd_frame)


                    detected_yolo_objects = self.get_results(yolo_results, self.custom_tracker, upd_frame)
                    counter, targets = self.count_objects(detected_yolo_objects, self.yolo_names)

                    frame, _ = self.draw(detected_yolo_objects, frame, self.yolo_names, targets)

            self.display_objects(counter, frame)
            cv2.imshow('YOLO Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

path = "/Users/maxkucher/opencv/howitzer_detector/video_6.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
model_path = "/Users/maxkucher/opencv/howitzer_detector/deepsort/artillery_detecor.pt"
yolo_model_path = "yolo11l.pt"

tracker = DeepDetector(path, device, 0.4, model_path, yolo_model_path)
tracker()


