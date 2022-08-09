import tflite_runtime.interpreter as interpreter
import numpy as np
import RPi.GPIO as GPIO
import cv2 as cv
from threading import Thread
from picamera2 import Picamera2

class VideoStream:
    def __init__(self):
        self.piCam = Picamera2()
        self.piCam.configure(self.piCam.create_preview_configuration())
        self.piCam.start()
        self.frame = []
        self.stopEx = False

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
        labels = 'labelmap.txt'
        model_path = 'detect.tflite'
        self.stream = stream
        self.model = interpreter.Interpreter(model_path=model_path,num_threads=4)

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        input_tensor_index = self.input_details[0]['index']

        self.mean = 127.5
        self.std = 127.5

        self.frame = []
        while len(self.frame) == 0:
            self.frame = stream.read()
        self.imgW = self.frame.shape[1]
        self.imgH = self.frame.shape[0]
        self.imgW_resize = 300 #frame.shape[1]
        self.imgH_resize = 300 #frame.shape[0]

        # config = np.array([1,self.imgH_resize,self.imgW_resize,3],dtype=np.int32)
        # model.resize_tensor_input(input_tensor_index, config)
        self.model.allocate_tensors()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        self.confidence_thresh = 0.6
        self.boxes_id, self.classes_id, self.scores_id = 0, 1, 2


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
            
            for i in range(4):
                if scores[i] < self.confidence_thresh or scores[i] > 1.0:
                    continue
                ymin = int(max(1,self.imgH*boxes[i][0]))
                xmin = int(max(1,self.imgW*boxes[i][1]))
                ymax = int(min(self.imgH,self.imgH*boxes[i][2]))
                xmax = int(min(self.imgW,self.imgW*boxes[i][3]))
                cv.rectangle(self.frame,(xmin,ymin),(xmax,ymax),(255,0,0),3)

            cv.imshow('detected',cv.cvtColor(self.frame,cv.COLOR_RGB2BGR))

            if cv.waitKey(1) & 0xFF == 27:
                self.stream.stop()

if __name__ == '__main__':
    stream = VideoStream().start()
    detect = Detect(stream=stream).start()