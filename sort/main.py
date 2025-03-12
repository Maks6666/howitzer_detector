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

        self.tracker = Sort(max_age=1000, min_hits=1, iou_threshold=0.4)
        self.hofitzer_names = self.hof_model.names
        self.yolo_names = self.det_model.names



    def load_hof_model(self):
        model = YOLO("/Users/maxkucher/opencv/howitzer_detector/hofitzer_detector.pt")
        model.to(self.device)
        model.fuse()
        return model

    def load_det_model(self):
        model = YOLO("yolo11l.pt")
        model.to(self.device)
        model.fuse()
        return model

    def get_hof_results(self, frame):
        resutls = self.hof_model.predict(frame, verbose=True, conf = 0.2)
        return resutls

    def get_det_results(self, frame):
        resutls = self.det_model.predict(frame, verbose=True, conf = 0.3)
        return resutls

    def howitzers(self, resutls):
        res_arr = []
        for result in resutls[0]:
            # print(type(f"Hkjkk {result}"))
            bboxes = result.boxes.xyxy.cpu().numpy()
            score = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy()

            res = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], score[0], class_id[0]]
            res_arr.append(res)

        return np.array(res_arr)

    def detect_objects(self, results):
        res_arr = []
        for result in results[0]:
            # print(type(f"Okooj {result}"))
            bboxes = result.boxes.xyxy.cpu().numpy()
            score = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy()

            arr = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], score[0], class_id[0]]
            res_arr.append(arr)

        return np.array(res_arr)


    def draw_hofitzer_frame(self, frame, bboxes, idc, classes):
        for bbox, idx, cls in zip(bboxes, idc, classes):

            text = f"[{idx}]:{self.hofitzer_names[int(cls)]}"

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return frame

    def draw_det_frame(self, frame, bboxes, idc, classes):
        for bbox, idx, cls in zip(bboxes, idc, classes):
            text = f"{idx}:{self.yolo_names[int(cls)]}"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            # print(type(frame))

            if not ret:
                break

            hofitzer_results = self.get_hof_results(frame)
            res_array = self.howitzers(hofitzer_results)



            if len(res_array) == 0:
                res_array = np.empty((0, 5))
                # continue
            # else:

            res = self.tracker.update(res_array)

            bboxes = res[:, :-1]
            # print(f"bboxes: {bboxes}")
            idc = res[:, -1].astype(int)
            classes = res_array[:, -1].astype(int)

            frame = self.draw_hofitzer_frame(frame, bboxes, idc, classes)
            # cv2.imshow('frame', frame)
            if len(bboxes) > 0:
                x1 = int(bboxes[0][0])
                y1 = int(bboxes[0][1])
                x2 = int(bboxes[0][2])
                y2 = int(bboxes[0][3])

                new_x1 = max(0, x1 - (x2 - x1) // 0.5)
                new_y1 = max(0, y1 - (y2 - y1) // 0.5)
                new_x2 = min(frame.shape[1], x2 + (x2 - x1) // 0.5)
                new_y2 = min(frame.shape[0], y2 + (y2 - y1) // 0.5)
                cropped_frames = frame[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
                # print(type(cropped_frames))

                # cv2.rectangle(frame, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (0, 0, 255), 2)
                detected_objects = self.get_det_results(cropped_frames)
                # print(detected_objects)
                upd_res_arr = self.detect_objects(detected_objects)

                if len(upd_res_arr) == 0:
                    upd_res_arr = np.empty((0, 5))

                upd_res = self.tracker.update(upd_res_arr)

                upd_bboxes = upd_res[:, :-1]
                print(upd_bboxes)
                upd_idc = upd_res[:, -1].astype(int)
                upd_classes = upd_res_arr[:, -1].astype(int)

                frame = self.draw_det_frame(frame, upd_bboxes, upd_idc, upd_classes)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

file = "/Users/maxkucher/opencv/howitzer_detector/video_4.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
tracker = Tracker(file, device)
tracker()





