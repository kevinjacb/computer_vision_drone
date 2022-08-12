import tflite_runtime.interpreter as interpreter
import numpy as np
import RPi.GPIO as GPIO
import cv2 as cv
import sys
import time
import math
import matplotlib.pyplot as plt
import smbus
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
    # def start(self):                                   #to run on a separate thread
    #     Thread(target=self.update,args=()).start()
    #     return self

    # def update(self):
    #     while not self.stopEx:
    #         self.frame = self.piCam.capture_array()[:,:,:3]

    def update(self):
        self.frame = self.piCam.capture_array()[:,:,:3]
    
    def getInstance(self):
        return self

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

        self.stopped = False

        self.label = ''
        with open(labels,'r') as f:
            self.label = f.read()

        self.label = self.label.split('\n')
        if self.label[0] == '???':
            del(self.label[0])
    def getInstance(self):
        return self
    def start(self,poly=None):
        self.stopped = False
        self.ct = CentroidTracker()
        Thread(target=self.detect,args=(poly,)).start()

    def isIn(self,rects,points):
        for i,(xmin,ymin,xmax,ymax) in enumerate(rects):
            for x,y in points:
                if x >= xmin and x <= xmax:
                    if y >= ymin and y <= ymax:
                        continue
                    else:
                        break
                else:
                    break
            return i
        return -1
    def detect(self,poly=None):
        global is_tracking,bbox_coordinates
        self.poly = poly
        locked_on = False
        if self.poly == None:
            is_tracking = False
            triggerDetection()

        global frames
        id = -1
        while not self.stopped:
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
            for i in scores_sorted[-4:]:
                if (scores[i] < self.confidence_thresh or scores[i] > 1.0) and self.label[i] != 'person':
                    continue
                ymin = int(max(1,imgH*boxes[i][0]))
                xmin = int(max(1,imgW*boxes[i][1]))
                ymax = int(min(imgH,imgH*boxes[i][2]))
                xmax = int(min(imgW,imgW*boxes[i][3]))
                if id == -1:
                    cv.rectangle(self.frame,(xmin,ymin),(xmax,ymax),(255,0,0),3)
                d_rects.append([xmin,ymin,xmax,ymax])

            if id == -1:
                id = self.isIn(d_rects,self.poly)
                if id == -1:
                    is_tracking = False
                    triggerDetection()

            objects = self.ct.update(rects=d_rects)
            
            # if locked_on and id not in list(objects.keys()):
            #     is_tracking = False
            #     triggerDetection()

            if not locked_on and id != -1:
                id = int(list(objects.keys())[id])
                locked_on = True
            
            else:
                try:
                    centroid = objects[id]
                    text = "Tracking tis idiot {}".format(id)
                    cv.putText(self.frame, text, (centroid[0] - 10, centroid[1] - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.circle(self.frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                except:
                    is_tracking = False
                    triggerDetection()
            

            rect_id = self.isIn(d_rects,list([objects[id]]))
            cv.rectangle(self.frame,d_rects[rect_id][:2],d_rects[rect_id][-2:],(255,0,0),3)
            bbox_coordinates = (d_rects[rect_id][:2],d_rects[rect_id][-2:])
            # if not self.is_tracking:
            #     pass                        #TODO

            frames['detection'] = self.frame

    def stop(self):
        self.stopped = True

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
            frames['detection'] = frame

        
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

class PID:
    def __init__(self):
        global imgW, imgH,prev_box_mid
        self.kp = 1
        self.kd = 0.5
        self.ki = 0.01
        self.center = prev_box_mid = [imgW//2,imgH//2]


    def calcPID(self):
        global bbox_coordinates,prev_box_mid,curr_mid
        curr_mid = ((bbox_coordinates[0][0]+bbox_coordinates[1][0])/2,(bbox_coordinates[0][1]+bbox_coordinates[1][1])/2)
        errorX = self.center[0] - curr_mid[0]
        dx = curr_mid[0] - prev_box_mid[0]
        pidPX = int(self.kp*errorX)
        pidDX = int(self.kd*dx)
        pidIX = 0
        if abs(curr_mid[0] - self.center[0]) < 50:
            pidIX = int(self.ki*errorX)
        prev_box_mid = curr_mid
        return pidPX + pidDX + pidIX
    

def triggerDetection():
    global detect,pdetect,is_tracking
    if is_tracking:
        poly = pdetect.getRect()
        pdetect.stop()
        detect.start(poly)  
    else:
        detect.stop()
        pdetect.start()

def read_from_arduino():
    global data,data_available
    try:
        data = bus.read_i2c_block_data(ADDR,0,30)
        data = [chr(s) for s in data]
        data = ''.join(data).split('#')
        data = data[1:-1]
        print(data)
        data = [int(x) for x in data]
        data_available = True
    except:
        print('An error has occurred')
        data_available = False

def write_to_arduino(data):
    data_str = '#'.join(map(str,data))
    data = list(bytes(data_str,'utf-8'))
    print(data_str,data)
    # try:
    bus.write_i2c_block_data(ADDR, 0, data)
    # except:
    #     print('error')

def isr(channel):                                           #ERROR gets called unnecessarily
    global pdetect,detect,data_available,is_tracking
    print('#########################asdfs############################')               
    if GPIO.input(channel):
        ctr = 0
        while not data_available and ctr < 10:
            read_from_arduino()
            ctr+=1
        if data_available:
            triggerDetection()
            write_to_arduino([1])
    else:
        data_available=False
        pdetect.stop()
        detect.stop()
        is_tracking=False
        write_to_arduino([0])

######################### without external mcu

# def calcPWM(channel):
#     global pwm_vals,pwm_counts
#     if GPIO.input(channel):
#         started = time.time()
#     else:
#         pulse_width = time.time()-started
#         pwm_vals[channel] += pulse_width
#         pwm_counts[channel] += 1

# def getPWM():
#     global pwm_in,pwm_counts
#     GPIO.add_event_detect(pwm,GPIO.BOTH,calcPWM)
#     sleep(3)
#     GPIO.remove_event_detect(pwm)
#     if 0 in pwm_counts.values():
#         return False
#     pwm_vals = {x:float(pwm_vals[x]/pwm_counts[x]) for x in list(pwm_vals.keys())}
#     print(pwm_vals)
#     return True
#pins normally high -> 3,5,7,24,26
#pins 13 and 15
# switch_pin = 3
# pwm_in = (10,11,12,13,15)  #pins for reading pwm signals -> (aileron, rudder)
# pwm_out = (5,7,24,26,16)
# pwm_vals = {10:1000,11:1000,12:1000,13:1000,15:1000}
# pwm_counts = {10:0,11:0,12:0,13:0,15:0}

#########################
ADDR = 9
interrupt = 7

imgW = imgH = 0
is_tracking = False
frames = dict({'detection' : np.ones(shape=(640,480,3),dtype=np.float32)})

GPIO.setmode(GPIO.BOARD)
GPIO.setup(interrupt,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.add_event_detect(interrupt,GPIO.BOTH,isr)

bus = smbus.SMBus(1)
######################## without external mcu

# GPIO.setup(pwm_in,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
# GPIO.setup(pwm_out,GPIO.OUT)
# GPIO.output(pwm_out,GPIO.LOW)
########################

data_available = False #input pwm values from arduino
data=[0] #format - option, rudder, elevator, aileron, gps select

bbox_coordinates = [[0,0],[0,0]]
prev_box_mid = (0,0)

stream = VideoStream().getInstance()
stream.update()
time.sleep(1)
pdetect = PoseDetection(stream=stream).getInstance()
detect = Detect(stream=stream).getInstance()
pid = PID()
# print(detect,pdetect)


while True:
    stream.update()
    cv.imshow('detected',cv.cvtColor(frames['detection'],cv.COLOR_BGR2RGB))

    if(data_available):
        Pid = pid.calcPID()
        print(Pid)
        dup_data = data.copy()
        dup_data[1] += Pid
        write_to_arduino(data)
    # print(is_tracking)
    if cv.waitKey(10) & 0xFF == 27 :
        stream.stop()
        pdetect.stop()
        detect.stop()
        break
GPIO.cleanup()
cv.destroyAllWindows()