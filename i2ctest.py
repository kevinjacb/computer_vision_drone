import smbus
import time

bus = smbus.SMBus(1)

while True:
    try:
        data = (bus.read_i2c_block_data(9,0,30))
        data = [chr(s) for s in data ]
        data_bytes = list(bytes('recieved','utf-8'))
        bus.write_i2c_block_data(9, 0,data_bytes)
    except:
        print('failed to connect, trying again')
    print(data,'\n sent :',data_bytes)
    time.sleep(0.1)
    