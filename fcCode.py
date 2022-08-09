import tflite_runtime.interpreter as interpreter
import numpy as np
import RPi.GPIO as GPIO
import cv2 as cv
import sys
import time
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
        imgW,imgH = self.piCam.capture_array().shape[:2]

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while not self.stopEx:
            self.frame = self.piCam.capture_array()[:,:,:3]

    def read(self):
        return np.array(self.frame.copy())

    def stop(self):
        self.stopEx = True
    
class Detect:
    def __init__(self, stream):
        global imgW,imgH
        labels = 'labelmap.txt'
        model_path = 'detect.tflite'
        
        self.is_tracking = False                    #TODO

        self.stream = stream
        self.model = interpreter.Interpreter(model_path=model_path,num_threads=4)

        self.ct = CentroidTracker()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        input_tensor_index = self.input_details[0]['index']

        self.mean = 127.5
        self.std = 127.5

        self.frame = []
        while len(self.frame) == 0:
            self.frame = stream.read()

        self.imgW_resize = 300 #frame.shape[1]
        self.imgH_resize = 300 #frame.shape[0]

        # config = np.array([1,self.imgH_resize,self.imgW_resize,3],dtype=np.int32)
        # model.resize_tensor_input(input_tensor_index, config)
        self.model.allocate_tensors()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        self.confidence_thresh = 0.6
        self.boxes_id, self.classes_id, self.scores_id = 0, 1, 2

        self.label = ''
        with open(labels,'r') as f:
            self.label = f.read()

        self.label = self.label.split('\n')
        if self.label[0] == '???':
            del(self.label[0])

    def start(self):
        Thread(target=self.detect,args=()).start()
        return self

    def detect(self):
        while not self.stream.stopEx:
            self.frame = self.stream.read()
            frame_inp = self.frame.copy()
            frame_inp = cv.resize(frame_inp,(self.imgW_resize,self.imgH_resize),cv.INTER_AREA)
            if self.input_details[0]['dtype'] == np.float32:
                frame_inp = (frame_inp - self.mean)/self.std
            frame_inp = np.expand_dims(frame_inp,axis=0)
            
            self.model.set_tensor(self.input_details[0]['index'],frame_inp)
            self.model.invoke()

            boxes = self.model.get_tensor(self.output_details[self.boxes_id]['index'])[0]
            classes = self.model.get_tensor(self.output_details[self.classes_id]['index'])[0]
            scores = self.model.get_tensor(self.output_details[self.scores_id]['index'])[0]

            scores_sorted = np.argsort(scores,axis=0)
            
            d_rects = []
            # print(scores_sorted)
            for i in scores_sorted[-5:]:
                if (scores[i] < self.confidence_thresh or scores[i] > 1.0) and self.label[i] != 'person':
                    continue
                ymin = int(max(1,imgH*boxes[i][0]))
                xmin = int(max(1,imgW*boxes[i][1]))
                ymax = int(min(imgH,imgH*boxes[i][2]))
                xmax = int(min(imgW,imgW*boxes[i][3]))
                cv.rectangle(self.frame,(xmin,ymin),(xmax,ymax),(255,0,0),3)
                d_rects.append([xmin,ymin,xmax,ymax])

            objects = self.ct.update(rects=d_rects)
            
            for objId, centroid in objects.items():
                text = "ID {}".format(objId)
                cv.putText(self.frame, text, (centroid[0] - 10, centroid[1] - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.circle(self.frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
            if not is_tracking:
                pass                        #TODO

            cv.imshow('detected',cv.cvtColor(self.frame,cv.COLOR_RGB2BGR))

            if cv.waitKey(1) & 0xFF == 27:
                self.stream.stop()

class PoseDetection:
    def __init__(self,stream):
        global imgW,imgH
        self.stream = stream
        
        model_path = 'pose.tflite'
        self.model = interpreter.Interpreter(model_path=model_path,num_threads=4)
        self.model.allocate_tensors()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

        self.imgH_resize,self.imgW_resize = self.input_details[0]['shape_signature'][1:3]
        print(self.imgH_resize,self.imgW_resize)
        print('\n')
        print(imgW,imgH)
        # print(self.input_details)
        # print('\n')
        # print(self.output_details)

        Thread(target=self.getPose,args=()).start()

    def getPose(self):
        while True:
            frame = self.stream.read()
            frame_inp = cv.resize(frame,(self.imgH_resize,self.imgW_resize),interpolation=cv.INTER_AREA)
            frame_inp = np.array(np.expand_dims(frame_inp,axis=0),dtype=np.float32)

            self.model.set_tensor(self.input_index,frame_inp)
            self.model.invoke()

            keypoints = self.model.get_tensor(self.output_index)[0][0]
            for keypoint in keypoints:
                if keypoint[2] < 0.3:
                    continue
                cv.circle(frame,(int(imgH*keypoint[1]),int(imgW*keypoint[0])),4,(255,0,0),-1)

            cv.imshow('pose',frame)
            if cv.waitKey(10) & 0xFF == 27:
                cv.destroyAllWindows()
                return
        # sys.exit()

class PID:
    def __init__(self):
        global imgW, imgH
        self.kp = 1
        self.kd = 0.5
        self.ki = 0.01
        self.center = [imgW//2,imgH//2]

    def calcPID(self,centroid,prevCentroid):
        error = self.center[0] - centroid[0]
        dx = centroid[0] - prevCentroid[0]
        pidP = int(self.kp*error)
        pidD = int(self.kd*dx)
        if abs(centroid[0] - self.center[0]) < 50:
            pidI = int(self.kI*error)
        
        return pidP + pidD + pidI
    

imgW = imgH = 0
stream = VideoStream().start()
time.sleep(1)
# detect = Detect(stream=stream).start()

PoseDetection(stream=stream)