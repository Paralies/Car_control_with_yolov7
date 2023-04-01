from pid import PID
from server import *
from image_processing import imgProcessing

import cv2
import numpy as np
import math
import argparse

#(Edit 0.0.1) From yolov7 by paralies
from numpy import random 
import torch
import time

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
#(Edit 0.0.1) From yolov7 by paralies

CAMERA_PORT = 2210
VEHICLE_INFO_PORT = 50301
LIDAR_PORT = 50001
CONTROL_SIGNAL_PORT = 50302

parser = argparse.ArgumentParser(description="\nDongDongMotors Autonomous Drive System(ADS) Version 3.0.0")
parser.add_argument('--ipg', help="IPG-IP")
parser.add_argument('--host', help="Host-IP")

args = parser.parse_args()

IPG_IP = args.ipg
HOST_IP = args.host

if IPG_IP is None:
    IPG_IP = '127.0.0.1'

if HOST_IP is None:
    HOST_IP = '127.0.0.1'

print('Target IPG IP:', IPG_IP, 'Target HOST IP', HOST_IP)

prev_frame_t = time.time()
p_carData = np.zeros((720, 1280), np.uint8)
    
tcp_camera = CameraReceiver(IPG_IP, CAMERA_PORT)
    
udp_vehicleInfo = VehicleInfoReceiver(HOST_IP, VEHICLE_INFO_PORT)

udp_lidar = LidarDataReceiver(HOST_IP, LIDAR_PORT)
    
udp_controlSignal = ControlSignalSender(IPG_IP, CONTROL_SIGNAL_PORT)
    
read_socket_list = [tcp_camera.get_socket(), udp_vehicleInfo.get_socket(), udp_lidar.get_socket()]

img_processor = imgProcessing()

accel_PID = PID(1 / 100, 0.0001, 0.001)
brake_PID = PID(1 / 200, 0.0001, 0.001)
stear_PID = PID(3.5, 0.1, 0.1)

accel_q = [ 0 for _ in range(640) ]
brake_q = [ 0 for _ in range(640) ]
stear_q = [ 0 for _ in range(640) ]

accel = 0
brake = 1
gear = 0
stear = 0

shouldStop = True

#===================(Edit 0.0.1) Set device for car detection by paralies===================
#Initialize
set_logging()
device = select_device()
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(['yolov7.pt'], map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

model = TracedModel(model, device, 640)

if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1

logFile = open('found.log', 'w')

view_img = True # Should be edited in challenge day!!!

print("YOLOv7 model set done!")
#===================(Edit 0.0.1) END Set device for car detection by paralies===================
    
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
        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        #Calculate and draw error
        hud, error = img_processor.draw_error(src, hsv, int(300 + brake * 20), 200)
        #Show result
        cv2.imshow("HUD", hud)

        hsv_mask = img_processor.hsv2binary(hsv)
        cv2.imshow('Line', hsv_mask)

        hsv_mask = img_processor.getNotRoad(hsv)
        cv2.imshow('NotRoadArea', hsv_mask)
        
        #cv2.imshow("White", img_processor.getWhiteCurb(hsv))
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

        #===================(Edit 0.0.1) Car detection with yolov7 by paralies===================
    try:
        yoloSrc = src
        t0 = time.time()

        # Padded resize
        img = letterbox(yoloSrc, 640, stride=32)[0] # Not the code included in detect.py

        # Convert
        
        # # 2023.03.24 0.0.2ver.try
        # src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        # # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img.transpose(2, )
        # img = np.ascontiguousarray(img)
        # # 2023.03.24 0.0.2ver.try
        
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416 # Not the code included in detect.py
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # print(img.shape)

        # Warmup
        # If the size of the image is changed, compared to the prvious size of the image, then warm-up again! -> warm-up could be pre-executed before the detection
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            #Dimension should be edited!!!
            print("checkpoint1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # https://discuss.pytorch.org/t/dimensions-of-an-input-image/19439/5
            # torch input dimension => (N, C, H, W)
            # While array uses the format as (H, W, C)
            for i in range(3):
                model(img, augment=False)[0]
            print("checkpoint2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred)
        t3 = time_synchronized()

        # print(pred)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(yoloSrc.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], yoloSrc.shape).round()

                # Get label results
                save_conf = False
                found = list()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    found.append(line)
                    
                    logFile.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, yoloSrc, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            # print(f'({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            '''
            COCO Label:
            PERSON = 0
            BICYCLE = 1
            CAR = 2
            MOTORBIKE = 3
            AEROPLANE = 4
            BUS = 5
            TRAIN = 6
            TRUCK = 7
            BOAT = 8
            TRAFFIC_LIGHT = 9
            FIRE_HYDRANT = 10
            STOP_SIGN = 11
            PARKING_METER = 12
            '''

            # Stream results
            if view_img:
                cv2.imshow("Car Detection", yoloSrc)

    except Exception as e:  
        print(e)
        break
    #===================(Edit 0.0.1) END Car detection with yolov7 by paralies===================
        
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

logFile.close() # by paralies