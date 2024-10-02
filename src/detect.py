#!/usr/bin/env python3

import cv2
import rospy
from cob_perception_msgs.msg import Detection, DetectionArray
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from ultralytics import YOLO
import os
import numpy as np
import torch
import time

class Detector:
    def __init__(self):

        ## CUDA usage 
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        # rospy.loginfo(f'[device]:{self.device}')
        rospy.sleep(2)

        ## Model CheckPoint
        model_path = "/home/stride_ws/src/stride_utils/vision_utils/src/train_n_final/weights/best.pt"
        self.model = YOLO(model_path)

        self.image_sub = rospy.Subscriber(
            "/image_jpeg_traffic/compressed", CompressedImage, self.callback, queue_size=1
        )

        self.cob_detection_pub_result = rospy.Publisher(
            "/yolov10/detection_results", DetectionArray, queue_size=1
        )

        self.cob_detection_pub_visualization = rospy.Publisher(
            "/image_jpeg_visualization/image", Image, queue_size=1
        )

        ## Parse Category_Labels
        names = ['none', 'background', 'green', 'red', 'red_left', 'red_yellow', 'yellow']

        self.label_to_class_dict = {0.0:names[0], 
                                    1.0:names[1], 
                                    2.0:names[2], 
                                    3.0:names[3], 
                                    4.0:names[4], 
                                    5.0:names[5], 
                                    6.0:names[6]}
        
        self.class_to_color_dict = {names[0]:(255, 255, 255), 
                                    names[1]:(0, 0, 0), 
                                    names[2]:(0, 255, 0), 
                                    names[3]:(0, 0, 255), 
                                    names[4]:(144, 238, 144), 
                                    names[5]:(0, 165, 255),
                                    names[6]:(0, 255, 255)}

        self.integrated_color_dict = {names[2]:names[2],  # green -> green
                                      names[3]:names[3],  # red -> red
                                      names[4]:names[3],  # red_left -> red
                                      names[5]:names[3],  # red_yellow -> red
                                      names[6]:names[6]}  # yellow -> yellow

        self.bridge = CvBridge()

        # ## info
        # rospy.loginfo(f"[initalized path] :{os.getcwd()}")
        # rospy.loginfo(f"[initalized model path] : {model_path}")
        # rospy.loginfo("yolov10 traffic light detection node initiated.\n\n")

    def callback(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # tik = time.perf_counter()
        res = self.model(im, verbose=False, device=self.device)[0]
        # tok = time.perf_counter()
        # rospy.loginfo(f'[inference time]:{tok-tik}s')
        
        detection_array = DetectionArray()
        detection_array.header = data.header

        if len(res.boxes):
            # tik = time.perf_counter()
            # if len(res.boxes) == 1:
            #     log = f'{len(res.boxes)} object detected.'
            # else:
            #     log = f'{len(res.boxes)} objects detected.'
            # rospy.loginfo(log)
            
            for box in res.boxes:
                ## Parse bbox data
                cls = str(self.label_to_class_dict[box.cls.item()])

                ## Ignore none, background class
                if cls in ['none', 'background']:
                    continue

                conf_score = box.conf.item()

                x_min = int(box.xyxy[0][0].item())
                y_min = int(box.xyxy[0][1].item())
                x_max = int(box.xyxy[0][2].item())
                y_max = int(box.xyxy[0][3].item())
                x_center = box.xywh[0][0].item() # float
                y_center = box.xywh[0][1].item() # float
                weidth = int(box.xywh[0][2].item())
                height = int(box.xywh[0][3].item())

                detection = Detection()
                # cob_perception_msgs/Detection Message
                detection.header = data.header
                detection.label = self.integrated_color_dict[cls]
                # detection.id = int(box.id)
                detection.detector = "yolov10_detection_result"
                detection.score = conf_score

                # xywh
                detection.mask.roi.x = x_min
                detection.mask.roi.y = y_min
                detection.mask.roi.width = weidth
                detection.mask.roi.height = height

                # xy
                detection.bounding_box_lwh.x = x_center
                detection.bounding_box_lwh.y = y_center

                detection_array.detections.append(detection)

                ############## visualization ##############
                # 박스 그리기: 이미지, 좌상단 좌표, 우하단 좌표, 색상(BGR), 두께
                cv2.rectangle(im, (x_min, y_min), (x_max, y_max), self.class_to_color_dict[cls], 2)
                
                # 텍스트 그리기: 이미지, 텍스트, 좌표, 폰트, 크기, 색상, 두께
                cv2.putText(im, cls, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.class_to_color_dict[cls], 2)
            # tok = time.perf_counter()
            # rospy.loginfo(f'[visualization time]:{tok-tik}s')
                ############################################
    
        img_vis = self.bridge.cv2_to_imgmsg(im, encoding="passthrough")
        self.cob_detection_pub_result.publish(detection_array)
        self.cob_detection_pub_visualization.publish(img_vis)


if __name__ == "__main__":

    rospy.init_node("yolov10_detection", anonymous=True)
    detector = Detector()

    rospy.spin()