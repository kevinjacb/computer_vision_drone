from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2 as cv
from picamera2 import Picamera2

piCam = Picamera2()
piCam.configure(piCam.create_preview_configuration())
piCam.start()

labels = 'labelmap.txt'
model_path = 'detect.tflite'

model = Interpreter(model_path=model_path,num_threads=4)

input_details = model.get_input_details()
output_details = model.get_output_details()

input_tensor_index = input_details[0]['index']

boxes_id = 0
classes_id = 1

mean = 127.5
std = 127.5

frame = piCam.capture_array()[:,:,:3] 

imgW_resize = 300 #frame.shape[1]
imgH_resize = 300 #frame.shape[0]
imgW = frame.shape[1]
imgH = frame.shape[0]
config = np.array([1,imgH_resize,imgW_resize,3],dtype=np.int32)
# model.resize_tensor_input(input_tensor_index, config)
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

confidence_thresh = 0.6
boxes_id, classes_id, scores_id = 0, 1, 2

while True:
    frame = np.array(piCam.capture_array()[:,:,:3])
    frame_inp = frame.copy()
    frame_inp = cv.resize(frame_inp,(imgW_resize,imgH_resize),cv.INTER_AREA)
    if input_details[0]['dtype'] == np.float32:
        frame_inp = (frame_inp - mean)/std
    frame_inp = np.expand_dims(frame_inp,axis=0)
    
    model.set_tensor(input_details[0]['index'],frame_inp)
    model.invoke()

    boxes = model.get_tensor(output_details[boxes_id]['index'])[0]
    classes = model.get_tensor(output_details[classes_id]['index'])[0]
    scores = model.get_tensor(output_details[scores_id]['index'])[0]

    scores_sorted = np.argsort(scores,axis=0)
    
    for i in range(4):
        if scores[i] < confidence_thresh or scores[i] > 1.0:
            continue
        ymin = int(max(1,imgH*boxes[i][0]))
        xmin = int(max(1,imgW*boxes[i][1]))
        ymax = int(min(imgH,imgH*boxes[i][2]))
        xmax = int(min(imgW,imgW*boxes[i][3]))
        cv.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0),3)

    cv.imshow('detected',cv.cvtColor(frame,cv.COLOR_RGB2BGR))
    # print(boxes.shape,'\t',classes.shape,'\t',scores.shape)
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
print(input_details)
print('\n')
print(output_details)