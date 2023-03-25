from pid import PID
from server import *
from imageProcessing import imgProcessing

import cv2
import numpy as np
import math

IP = '127.0.0.1'
CAMERA_PORT = 2210
VEHICLE_INFO_PORT = 50301
LIDAR_PORT = 50001
CONTROL_SIGNAL_PORT = 50302

prev_frame_t = time.time()
p_carData = np.zeros((720, 1280), np.uint8)
    
tcp_camera = CamerReceiver(IP, CAMERA_PORT)
    
udp_vehicleInfo = VehicleInfoReceiver(IP, VEHICLE_INFO_PORT)

udp_lidar = LidarDataReceiver(IP, LIDAR_PORT)
    
udp_controlSignal = ControlSignalSender(IP, CONTROL_SIGNAL_PORT)
    
read_socket_list = [tcp_camera.get_socket(), udp_vehicleInfo.get_socket(), udp_lidar.get_socket()]

img_processor = imgProcessing()

accel_PID = PID(1 / 100, 0.0001, 0.001)
brake_PID = PID(1 / 200, 0.0001, 0.001)
stear_PID = PID(3, 0.01, 0.06)

accel_q = [ 0 for _ in range(640) ]
brake_q = [ 0 for _ in range(640) ]
stear_q = [ 0 for _ in range(640) ]

accel = 0
brake = 1
gear = 0
stear = 0

shouldStop = True
    
while True:
    conn_read_socket_list, conn_write_socket_list, conn_except_socket_list = select.select(read_socket_list, [], [])
    
    #Receive Data(Camera[TCP] / VehicleInfo[UDP] / Lidar[UDP]) with I/O Multiplexing
    for conn_read_socket in conn_read_socket_list:
        if conn_read_socket == tcp_camera.get_socket():
            #RECEIVE CAMERA DATA
            try:
                src, fps = tcp_camera.receive_img()
                font = cv2.FONT_HERSHEY_PLAIN
                text = 'FPS: ' + str(fps)
                cv2.putText(src, text ,(5,15),font, 1, (0,0,0), 2)
                cv2.imshow("Camera", src)
            except Exception as e:
                print(e)
        if conn_read_socket == udp_vehicleInfo.get_socket():
            #RECEIVE Vehicle Info DATA
            try:
                new_frame_t = time.time()
                if new_frame_t - prev_frame_t > 0.2:
                    carData = udp_vehicleInfo.receive_datas()
                    p_carData = udp_vehicleInfo.make_print(carData)
                    prev_frame_t = new_frame_t
                cv2.imshow("CarData", p_carData)
            except Exception as e:  
                print(e)
        if conn_read_socket == udp_lidar.get_socket():
            #Receive Lidar Data
            try:
                lidar_data = 1
            except Exception as e:  
                print(e)
        
    #Calculate Control Signal
    try:
        #HSV transform
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        #Calculate and draw error
        hud, error = img_processor.draw_error(src, hsv, 300, 200)
        #Show result
        cv2.imshow("HUD", hud)
        if shouldStop != True:
            #Get error
            accel_error, brake_error, stear_error = error 
            #Compute error to PID
            stear_radian = math.atan(stear_error/50)
            stear = stear_PID.computePID(stear_radian)
            accel = accel_PID.computePID(accel_error)
            brake = brake_PID.computePID(brake_error)
            #Put in queue
            accel_q.pop(0)
            brake_q.pop(0)
            stear_q.pop(0)
            accel_q.append(accel)
            brake_q.append(brake)
            stear_q.append(stear * 180 / math.pi)
        #Draw Monitor of each control signal 
        accel_monitor = img_processor.draw_monitor(accel_q)
        cv2.imshow("Accel", accel_monitor)
        brake_monitor = img_processor.draw_monitor(brake_q)
        cv2.imshow("Brake", brake_monitor)
        stear_monitor = img_processor.draw_monitor(stear_q)
        cv2.imshow("Stear", stear_monitor)
    except Exception as e:  
        print(e)
        
    #Send Control Singnal
    try:
        udp_controlSignal.sendSignal(accel, brake, gear, stear)
        monitor = udp_controlSignal.make_print(accel, brake, gear, stear)
        cv2.imshow("ControlSignal", monitor)
    except Exception as e:
        print(e)
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    if cv2.waitKey(1) & 0xFF == ord('s'):
        gear = 1
        shouldStop = False
            
cv2.destroyAllWindows()

tcp_camera.quit_receive()
udp_vehicleInfo.quit_receive()
udp_lidar.quit_receive()
udp_controlSignal.quit_receive()