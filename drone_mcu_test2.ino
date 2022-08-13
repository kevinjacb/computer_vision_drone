#include <PinChangeInterrupt.h>
#include <Servo.h>
#include <Wire.h>

const byte channel_pin[] = {4,7,8,12,13};
const byte output_pin[] = {3,5,6,9,10};
const byte trig_pin = 11;

volatile unsigned long rising_start[] = {0, 0, 0, 0, 0};
volatile long channel_length[] = {1000, 1000, 1000, 1000, 1000};

String received_data;
bool toggle = false;

void setup() {
  Wire.begin(0x09);
  Wire.onRequest(requestEvent);
  Wire.onReceive(receiveEvent);
  Serial.begin(57600);

  for(byte i : channel_pin)
    pinMode(i,INPUT);
  pinMode(trig_pin,OUTPUT);
  digitalWrite(trig_pin,LOW);

  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(channel_pin[0]), onRising0, CHANGE);
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(channel_pin[1]), onRising1, CHANGE);
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(channel_pin[2]), onRising2, CHANGE);
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(channel_pin[3]), onRising3, CHANGE);
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(channel_pin[4]), onRising4, CHANGE);
  
}

void processPin(byte pin) {
  uint8_t trigger = getPinChangeInterruptTrigger(digitalPinToPCINT(channel_pin[pin]));

  if(trigger == RISING) {
    rising_start[pin] = micros();
  } else if(trigger == FALLING) {
    channel_length[pin] = micros() - rising_start[pin];
  }
}

void onRising0(void) {
  processPin(0);
}

void onRising1(void) {
  processPin(1);
}

void onRising2(void) {
  processPin(2);
}

void onRising3(void) {
  processPin(3);
}

void onRising4(void) {
  processPin(4);
}

void receiveEvent(int h){
  received_data = "";
  while(Wire.available())
    received_data += (char)Wire.read();
}

void requestEvent(){
  String data = "";
  for(int val : channel_length){
    data+= val;
    data+="#";
  }
  Wire.write(data.c_str());
  Serial.print(data);
}

void loop() {
  if(channel_length[0] > 1250){
    digitalWrite(trig_pin,HIGH);
    Serial.print("rpi mode: ");
  }
  else{
    digitalWrite(trig_pin,LOW);
    Serial.print("normal mode: ");
  }

    
  Serial.println(received_data);
  delay(100);
}
