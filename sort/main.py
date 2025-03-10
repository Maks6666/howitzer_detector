from sort import Sort
from ultralytics import YOLO
import torch
import cv2
import numpy as np


class Tracker:
    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.hof_model = self.load_hof_model()
        self.det_model = self.load_det_model()
        self.tracker = Sort(max_age=10, min_hits=5, iou_threshold=0.4)
        self.hofitzer_names = self.hof_model.names
        self.yolo_names = self.det_model.names



    def load_hof_model(self):
        model = YOLO("/Users/maxkucher/opencv/howitzer_detector/hofitzer_detector.pt")
        model.fuse()
        return model

    def load_det_model(self):
        model = YOLO("yolo11l.pt")
        model.fuse()
        return model

    def get_hof_results(self, frame):
        resutls = self.hof_model.predict(frame, verbose=True, conf = 0.3)
        return resutls

    def get_det_results(self, frame):
        resutls = self.det_model(frame)
        return resutls

    def howitzers(self, resutls):
        res_arr = []
        for result in resutls[0]:
            bboxes = result.boxes.xyxy.cpu().numpy()
            score = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy()

            res = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], score[0], class_id[0]]
            res_arr.append(res)

        return np.array(res_arr)


    def draw_hofitzer_frame(self, frame, bboxes, idc, classes):
        for bbox, idx, cls in zip(bboxes, idc, classes):

            text = f"{idc}:{self.hofitzer_names[int(cls)]}"

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return frame

    def draw_det_frame(self, upd_frame, bboxes, idc, classes):
        for bbox, idx, cls in zip(bboxes, idc, classes):
            text = f"{idc}:{self.det_model[cls]}"
            cv2.rectangle(upd_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(upd_frame, text, (int(bbox[0]), int(bbox[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return upd_frame

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            hofitzer_results = self.hof_model(frame)
            res_array = self.howitzers(hofitzer_results)

            if len(res_array) == 0:
                res_array = np.empty((0, 5))

            res = self.tracker.update(res_array)

            bboxes = res[:, :-1]
            idc = res[:, -1].astype(int)
            classes = res_array[:, -1].astype(int)

            frame = self.draw_hofitzer_frame(frame, bboxes, idc, classes)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

file = "/Users/maxkucher/opencv/howitzer_detector/video_1.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tracker = Tracker(file, device)
tracker()




