import RPi.GPIO as GPIO
import time
# from machine import Pin,PWM

GPIO.setmode(GPIO.BOARD)


############################### PWM GENERATION
# GPIO.setup(7,GPIO.OUT)

# pwm = GPIO.PWM(7,50) #1000 - 520Hz
# start = 5
# step = 0.1
# pwm.start(start)
# time.sleep(2)
# print('starting motors here we gooooo')
# for j in range(10):
#     for i in range(20):
#         pwm.ChangeDutyCycle(start+i*step)
#         print(start+i*step)
#         time.sleep(1)

# pwm.stop()
################################
started = 0
pulse_width = 0
def calcPWM(channel):
    global started,pulse_width
    if GPIO.input(7):
        started = time.time()
    else:
        pulse_width = time.time()-started

GPIO.setup(7,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.add_event_detect(7,GPIO.BOTH,calcPWM)

delay = 0
while True:
    try:
        if(time.time() - delay > 1):
            print('pulse width -> ',pulse_width*1000000)
            delay = time.time()
        time.sleep(0.000001)
    except KeyboardInterrupt:
        print('stopping')
        break

GPIO.cleanup()

# pwm = PWM(Pin(7))
# pwm.freq(54)

# start = 3512
# step = 176

# while True:
#     for iter in range(20):
#         pwm.duty_u16(start+step*iter)
#         sleep(0.5)
# pwm.duty_u16(start)