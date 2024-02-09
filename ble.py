import asyncio #비동기적 프로그래밍 모듈
from bleak import BleakClient   #bleak 블루투스 클라이언트 모듈 import
import time

mac_address = "B0:A7:32:DE:BA:D2"   #BLE mac address 1번ESP:"A8:42:E3:47:6E:16" 2번ESP:"B0:A7:32:DC:E5:66"
notify_characteristic_uuid = "beb5483e-36e1-4688-b7f5-ea07361b26a8" #notify characteristic uuid

async def run(address): #구동함수
    character=0
    while True: #무한루프 동작  
        client = BleakClient(address) #BLE client 선언
        try:
            await client.connect()  #BLE connection 상태를 기다림
            print("Connected")  #연결완료되면 디버그구문 출력
            services = await client.get_services()  # Discover services and characteristics
            for service in services:    
                for char in service.characteristics:    #서비스를 검색하고 notify_characteristic_uuid와 일치하는것을 찾음.
                    if char.uuid == notify_characteristic_uuid:
                        try:
                            # Write data to the characteristic
                            with open('txt.txt', 'r') as f: #파일 열기
                                commend=f.read()    #commend 읽기
                        except:
                            commend="-1" #commend를 정상적으로 못 읽었을때 처리할 구문
                        commend_final=str(character).encode('utf-8')+b'\x2c'+commend.encode()
                        await client.write_gatt_char(char, bytearray(commend_final)) #실제 문자를 전송
                        print(commend_final) #쓴 문자열 출력
            # Continue running in the connected state
            while True:
                # Your additional logic 
                await asyncio.sleep(4)      #딜레이 조정하면 됨.
                if  client.is_connected:
                    if  character==0:
                        character=1
                        #print("1")
                    else:
                        character=0
                        #print("0")
                    services = await client.get_services()  # Discover services and characteristics
                    for service in services:    
                        for char in service.characteristics:    #서비스를 검색하고 notify_characteristic_uuid와 일치하는것을 찾음.
                            if char.uuid == notify_characteristic_uuid:
                                try:
                                    # Write data to the characteristic
                                    with open('txt.txt', 'r') as f: #파일 열기
                                        commend=f.read()    #commend 읽기
                                except:
                                    commend="-1" #commend를 정상적으로 못 읽었을때 처리할 구문
                                commend_final=str(character).encode('utf-8')+b'\x2c'+commend.encode()
                                await client.write_gatt_char(char, bytearray(commend_final)) #실제 문자를 전송
                                #print(commend_final) #쓴 문자열 출력
                
                else:   #연결이 끊긴경우는 while loop빠져나감
                    break
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await client.disconnect()   #BLE disconnect
            print("Disconnected")   #disconnect 디버그 출력
            await asyncio.sleep(5)  # Add a delay before attempting to reconnect

async def main():
    await run(mac_address)

asyncio.run(main())
