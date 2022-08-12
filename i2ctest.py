import smbus
import time


bus = smbus.SMBus(1)

while True:
    data=data_bytes = []

 
    data = (bus.read_i2c_block_data(9,0,32))
    data = [chr(s) for s in data ]
    data_bytes = list(bytes('recieved','utf-8'))
    bus.write_i2c_block_data(9, 0,data_bytes)

    print(data,'\n sent :',data_bytes)
    time.sleep(0.1)

GPIO.cleanup()