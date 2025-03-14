import cv2
from torchvision.ops import boxes
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch



class DeepDetector:
    def __init__(self, path, device, model_path, yolo_model_path):
        self.path = path
        self.device = device

        self.model = self.load_model(model_path)
        self.yolo_model = self.load_model(yolo_model_path)

        self.names = self.model.names
        self.yolo_names = self.yolo_model.names


        self.tracker = DeepSort(max_iou_distance = 0.7, max_age = 100, n_init = 2)

    def load_model(self, model_path):
        model = YOLO(model_path)
        model.to(self.device)
        model.fuse()
        return model

    # def load_yolo_model(self):
    #     model = YOLO("yolo11l.pt")
    #     model.to(self.device)
    #     model.fuse()
    #     return model

    def results(self, model, frame):
        return model(frame)[0]

    def get_results(self, results, frame):
        if len(results) !=0 and results is not None:
            res_array = []
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > 0.3:
                    res_array.append(([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)], float(score), int(class_id)))

                # print(res_array)
                tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)

                detected_objects = []
                for track in tracks:
                    bbox = track.to_ltrb()
                    idx = track.track_id
                    class_id = track.get_det_class()
                    detected_objects.append((bbox, idx, class_id))

                return detected_objects
        else:
            return None


    def draw(self, detected_objects, frame, names):
        if detected_objects is not None and len(detected_objects) > 0:
            for bbox, idx, class_id in detected_objects:
                x1, y1, x2, y2 = map(int, bbox)
                # bbox = (x1, y1, x2, y2)

                name = names[int(class_id)]
                text = f"{idx}:{name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # print(bbox)
            return frame, bbox
        else:
            return frame, None

    def detect_objects(self, frame, bbox):
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)

            new_x1 = max(0, x1 - (x2 - x1) // 0.5)
            new_y1 = max(0, y1 - (y2 - y1) // 0.5)
            new_x2 = min(frame.shape[0], x2 + (x2 - x1) // 0.5)
            new_y2 = min(frame.shape[0], y2 + (y2 - y1) // 0.5)
            # print(new_x1, new_x2, new_y1, new_y2)
            cv2.rectangle(frame, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (0, 0, 255), 2)
            upd_frame = frame[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
            return upd_frame

        else:
            return frame


    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # model = load_model("/Users/maxkucher/opencv/howitzer_detector/best.pt")
            # yolo_model = load_model("yolo11l.pt")

            results = self.results(self.model, frame)
            yolo_results = self.results(self.yolo_model, frame)

            detected_objects = self.get_results(results, frame)
            frame, bboxes = self.draw(detected_objects, frame, self.names)

            if bboxes is not None:
                upd_frame = self.detect_objects(frame, bboxes)

                if upd_frame is not None and upd_frame.size > 0:
                    detected_yolo_objects = self.get_results(yolo_results, frame)
                    frame, _ = self.draw(detected_yolo_objects, frame, self.yolo_names)

            cv2.imshow('YOLO Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

path = "/Users/maxkucher/opencv/howitzer_detector/video_4.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_path = "/Users/maxkucher/opencv/howitzer_detector/best.pt"
yolo_model_path = "yolo11l.pt"

tracker = DeepDetector(path, device, model_path, yolo_model_path)
tracker()



