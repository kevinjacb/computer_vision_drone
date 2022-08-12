import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)

def event(channel):
    if GPIO.input(7):
        print("PIO.input(7)")

GPIO.add_event_detect(7,GPIO.BOTH,event)

while True:
    pass

