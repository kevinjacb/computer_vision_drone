import tflite_runtime.interpreter as interpreter
import numpy as np
import RPi.GPIO as GPIO
import cv2 as cv
import sys
import time
import math
import matplotlib.pyplot as plt
from centroidtracker import CentroidTracker
from threading import Thread
from picamera2 import Picamera2
class VideoStream:
    def __init__(self):
        global imgW,imgH
        self.piCam = Picamera2()
        self.piCam.configure(self.piCam.create_preview_configuration())
        self.piCam.start()
        self.frame = []
        self.stopEx = False
        imgH,imgW = self.piCam.capture_array().shape[:2]
    # def start(self):
    #     Thread(target=self.update,args=()).start()
    #     return self
    def getInstance(self):
        return self

    def update(self):
        self.frame = self.piCam.capture_array()[:,:,:3]

    def read(self):
        return np.array(self.frame.copy())

    def stop(self):
        self.stopEx = True


class PoseDetection:  # 0 - jesus pose
    def __init__(self,stream):
        global imgW,imgH,is_tracking
        self.stream = stream
        
        model_path = 'pose.tflite'
        self.model = interpreter.Interpreter(model_path=model_path,num_threads=2)
        self.model.allocate_tensors()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

        self.stopped = False

        self.imgH_resize,self.imgW_resize = self.input_details[0]['shape_signature'][1:3]
        print(self.imgH_resize,self.imgW_resize)
        print('\n')
        print(imgW,imgH)
        # print(self.input_details)
        # print('\n')
        # print(self.output_details)
    def getInstance(self):
        return self

    def start(self):
        self.stopped = False
        Thread(target=self.getPose,args=()).start()

    def getRect(self):
        return self.rect

    def getPose(self):
        global frames,is_tracking
        while not self.stopped:
            frame = self.stream.read()
            frame_inp = cv.resize(frame,(self.imgH_resize,self.imgW_resize),interpolation=cv.INTER_AREA)
            frame_inp = np.array(np.expand_dims(frame_inp,axis=0),dtype=np.float32)

            self.model.set_tensor(self.input_index,frame_inp)
            self.model.invoke()

            keypoints = self.model.get_tensor(self.output_index)[0][0]
            self.rect = self.estimatePose(keypoints)
            if(self.rect != None):
                # print('detected ')
                is_tracking = True
                triggerDetection()

            for keypoint in keypoints:
                if keypoint[2] < 0.3:
                    continue
                cv.circle(frame,(int(imgW*keypoint[1]),int(imgH*keypoint[0])),4,(255,0,0),-1)
                frames['pose'] = frame
    def estimatePose(self,keypoints):
        points = np.arange(5,11)
        for point in points:
            if keypoints[point][2] < 0.4:
                return None
        dist_wrists = math.dist(keypoints[9][:2], keypoints[10][:2])
        dist_sum = math.dist(keypoints[5][:2],keypoints[6][:2])
        for i in range(2):
            dist_sum += math.dist(keypoints[5+i*2][:2],keypoints[5+(i+1)*2][:2])
            dist_sum += math.dist(keypoints[6+i*2][:2],keypoints[6+(i+1)*2][:2])
        if abs(dist_sum - dist_wrists) < dist_sum/7:
            return [keypoints[5][:2],keypoints[6][:2],keypoints[11][:2],keypoints[12][:2]]
        return None
        
    def stop(self):
        self.stopped=True

frames = dict({'pose': 0,'detection' : 1})
stream = VideoStream().getInstance()
stream.update()
time.sleep(1)
PoseDetection(stream).start()
while True:
    stream.update()
    cv.imshow('pose',frames['pose'])
    if cv.waitKey(10) & 0xFF == 27:
        break
cv.destroyAllWindows()