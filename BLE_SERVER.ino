#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// BLE 서버, 서비스, 특성 및 디스크립터 객체 선언
BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;
int data_no=-1;  //수신받는 상태값.
int pwm_pin=23; //pwm핀
int status_flag=0;

// BLE 서버 이름 설정
const char* BLE_DEVICE_NAME = "ESP32_Server";
// BLE 서비스 UUID 및 특성 UUID 설정
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

class MyServerCallbacks : public BLEServerCallbacks 
  {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("connected with BLE server");
    }
    void onDisconnect(BLEServer* pServer) 
    {
      deviceConnected = false;
      Serial.println("Disconnected with BLE server");
      // 연결이 끊어지면 다시 연결을 시도합니다.
      BLEDevice::startAdvertising();
    }
  };

void setup() {
  Serial.begin(115200);
  // BLE 초기화
  BLEDevice::init(BLE_DEVICE_NAME);
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  // BLE 서비스 생성
  BLEService* pService = pServer->createService(SERVICE_UUID);
  // BLE 특성 생성
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_WRITE |
                      BLECharacteristic::PROPERTY_NOTIFY |
                      BLECharacteristic::PROPERTY_INDICATE
                    );
  // BLE 특성 값 초기화
  pCharacteristic->setValue("-1");
  // BLE 특성 디스크립터 설정
  pCharacteristic->addDescriptor(new BLE2902());
  // BLE 서비스 시작
  pService->start();
  // BLE 광고 시작
  BLEAdvertising* pAdvertising = pServer->getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();
  Serial.println("BLE divice is initialized.");
  pinMode(pwm_pin,OUTPUT);
}

void loop() 
  {
    // 중앙 장치가 연결되었거나 연결이 해제되었을 때 처리
    if (deviceConnected != oldDeviceConnected) 
      {
        if (deviceConnected) 
          {
            Serial.println("connected with BLE server"); //for debug
            String readValue_a = pCharacteristic->getValue().c_str();
          } 
        else 
          {
            Serial.println("Disconnected with BLE server"); //for debug
          }
        oldDeviceConnected = deviceConnected;
      }
    
    // 중앙 장치와 연결된 경우 데이터 전송
    if (deviceConnected)
      {
        
        String readValue_a = pCharacteristic->getValue().c_str();
        int first=readValue_a.indexOf(",");
        int length_str=readValue_a.length();
            //Serial.println(readValue_a.substring(0,first));
            //Serial.println(readValue_a.substring(first+1,length_str));
            //Serial.println("Read value: " + readValue_a);
            // 여기서 수신된 데이터를 처리
        if  (readValue_a.substring(0,first).toInt()==status_flag)
          { 
            Serial.println(readValue_a.substring(first+1,length_str));
            if  (status_flag==1)
              {
                status_flag=0;
              }
            else
              {
                status_flag=1;
              }
            if (readValue_a.substring(first+1,length_str).toInt()<4 and readValue_a.substring(first+1,length_str).toInt()>=0) 
              {
                Serial.println("case1");
                for(int i=0; i<readValue_a.substring(first+1,length_str).toInt(); i++)
                {
                  digitalWrite(pwm_pin, HIGH);
                  delay(100);
                  digitalWrite(pwm_pin, LOW);
                  delay(50);
                }
              }
              else if(readValue_a.substring(first+1,length_str).toInt() >= 4)
                {
                  Serial.println("case2");
                  digitalWrite(pwm_pin, HIGH);
                  delay(300);
                  digitalWrite(pwm_pin, LOW);
                  delay(50);
                  for(int i=0; i<readValue_a.substring(first+1,length_str).toInt()-4; i++)
                  {
                    digitalWrite(pwm_pin,HIGH);
                    delay(100);
                    digitalWrite(pwm_pin,LOW);
                    delay(50);
                  }
                }
            }
      }
  }
