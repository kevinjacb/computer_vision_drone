#include <PinChangeInterrupt.h>
#include <Servo.h>
#include <Wire.h>

const byte channel_pin[] = {2,4,7,8,11};
const byte output_pin[] = {3,5,6,9,10};
const byte trig_pin = 12;

volatile unsigned long rising_start[] = {0, 0, 0, 0, 0};
volatile long channel_length[] = {1000, 1000, 1000, 1000, 1000},
 first_channel_length[] = {0,0,0,0,0};
volatile bool i2c_int[] = {false,false,false,false,false};

byte counter = 0;

String received_data;
bool toggle = false;

Servo output[5];

void setup() {
  Wire.setClock(400000);
  Wire.begin(0x09);
  Wire.onRequest(requestEvent);
  Wire.onReceive(receiveEvent);
  Serial.begin(115200);

  for(byte i : channel_pin)
    pinMode(i,INPUT);

  for(int i = 0; i < 5; i++){
    output[i].attach(output_pin[i]);
    output[i].writeMicroseconds(channel_length[i]);
  }
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
    if(i2c_int[pin]){
      i2c_int[pin] = false;
      return;
    }
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

void receiveEvent(int bytes){
//  noInterrupts();
  for(int i = 0; i < 5; i++)
    i2c_int[i] = true;
  received_data = "";
  byte ctr = 0;
  volatile long rc_length[5];
  while(Wire.available()){
    char data = Wire.read();
    if(received_data != '#')
      received_data += (char)Wire.read();
    else{
      rc_length[ctr++] = received_data.toInt();
      received_data = "";
    }
  }
  if(received_data.toInt() == 1){
    toggle = true;
    for(int i = 0; i < 5; i++)
      channel_length[i] = rc_length[i];
  }
  else
    toggle = false;
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
//  if(received_data.length() > 
  
  for(int i = 0; i < 5; i++)
    if(channel_length[i] > 990 && channel_length[i] < 2200)
      output[i].writeMicroseconds(channel_length[i]);
  for(int i : channel_length){
      Serial.print(i);
      Serial.print(" ");
    }
    Serial.println();
//  Serial.println(received_data);
  delay(20);
}
