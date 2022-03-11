import cv2
import numpy as np
from .yolo import darknet_person as darknet


class box_ext:
    def __init__(self,):
        self.yolo, self.class_name, _ = darknet.load_network(
            "/mnt/nas1/cody/cody_server5_home/deepvisions_trash_dump/data/datasets/yolo/yolov3-608.cfg",
            "/mnt/nas1/cody/cody_server5_home/deepvisions_trash_dump/data/datasets/yolo/coco.data",
            "/mnt/nas1/cody/cody_server5_home/deepvisions_trash_dump/data/datasets/yolo/checkpoints/yolov3-608.weights",
        )
        print("Yolo box crop Prepare......")

    def box_detect_crop(self, windows):
        ori_shape = windows[0].shape
        boxes_que = []
        for img in windows:
            img_resize = cv2.resize(img, (608, 608))
            detections = darknet.detect_image(
                self.yolo, self.class_name, img_resize, 0.5, 0.5
            )
            boxes = darknet.ext_only_boxes(detections)

            boxes_que.append(boxes)

        x1, y1, x2, y2 = self.crop_point_detect(boxes_que, ori_shape)

        if x1 == None:
            return windows

        else:
            x1 = int(x1 / 608 * ori_shape[1])
            x2 = int(x2 / 608 * ori_shape[1])
            y1 = int(y1 / 608 * ori_shape[0])
            y2 = int(y2 / 608 * ori_shape[0])

            crop_que = np.array(windows)[:, y1:y2, x1:x2, :]

            return crop_que

    def crop_point_detect(self, que, shape):
        aspect_ratio = shape[0] / shape[1]
        total_boxes = []
        for boxes in que:
            if boxes != []:
                total_boxes.extend(boxes)
        total_boxes = np.array(total_boxes)

        if len(total_boxes) > 3:
            track_x_min, track_y_min, _, _ = np.min(total_boxes, axis=0)
            _, _, track_x_max, track_y_max = np.max(total_boxes, axis=0)

            track_height = (track_y_max - track_y_min) * 1.4
            track_width = track_height / aspect_ratio

            track_x_cent = (track_x_max + track_x_min) / 2
            track_y_cent = (track_y_max + track_y_min) / 2

            track_x_min = max(0, int(track_x_cent - track_width / 2))
            track_y_min = max(0, int(track_y_cent - track_height / 2))
            track_x_max = min(608 - 1, int(track_x_cent + track_width / 2))
            track_y_max = min(608 - 1, int(track_y_cent + track_height / 2))

            return track_x_min, track_y_min, track_x_max, track_y_max
        else:
            return None, None, None, None

