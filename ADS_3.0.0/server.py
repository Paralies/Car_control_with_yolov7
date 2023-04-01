import socket
import cv2
import numpy as np
import struct
import time
import select

class TCP_Socket:
    def __init__(self, ip, port):
        self.IP = ip
        self.PORT = port
        self.address = (ip, port)
        self.socketOpen()
        
    def get_sock(self):
        return self.sock
        
    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.address)
        print('TCP Socket Connected[', self.IP, self.PORT, ']')
        
    def socketClose(self):
        self.sock.close()
        print('Socket Closed')
        
    def recvall(self, count):
        buf = b''
        while count:
            newbuf = self.sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    
    def recv(self, size):
        return self.sock.recv(size)

    
class CameraReceiver:
    def __init__(self, ip, port):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.address = (ip, port)
        self.tcp = TCP_Socket(ip, port)
        self.sock = self.tcp.get_sock()
        self.receive_metadata()
        
        self.prev_frame_t = 0
        
    def get_socket(self):
        return self.sock
        
    def receive_metadata(self):
        try:
            meta_data = self.tcp.recv(64)
            meta_data = meta_data.decode('utf-8')
            print(meta_data)

            data = self.tcp.recv(64)
            data = data.decode('utf-8')
            print(data)

            if len(data) == 64 and data[0] == '*':
                head = data.split(' ')
                img_type = head[2]
                self.width, self.height = map(int, head[4].split('x'))
                print('Detected WIDTH:', self.width, 'Detected HEIGHT:', self.height)
            else:
                print('mismatch metadata: ', len(data), 'was not matched with 64')
                print(len(data), str(data[0]))

            self.buf_size = self.width * self.height * 3
            trash_buffer = self.tcp.recv(59*64)
        except Exception as e:
            self.width = 0
            self.height = 0
            print(e)
            
    def receive_datas(self):
        try:
            buffer = self.tcp.recv(64)
            img = self.tcp.recvall(self.buf_size)
            img = np.fromstring(img, dtype = np.uint8)
            img = img.reshape(self.height, self.width, 3)
            
            new_frame_t = time.time()
            fps = 1 / (new_frame_t - self.prev_frame_t)
            self.prev_frame_t = new_frame_t
            return img, fps
        except Exception as e:
            print(e)
            return -1
    
    def receive_img(self):
        try:
            src, fps = self.receive_datas()
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            return src, fps
        except Exception as e:
            print(e)
            return -1
        
    def quit_receive(self):
        self.tcp.socketClose()
        
###--UDP--##############################

class UDP_Socket:
    def __init__(self, ip, port, socket_type):
        self.IP = ip
        self.PORT = port
        self.ADDRESS = (ip, port)
        self.socketOpen(socket_type)
        
    def get_sock(self):
        return self.sock
        
    def socketOpen(self, socket_type):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if socket_type == 'server': self.sock.bind(self.ADDRESS)
        print('UDP Socket Connected[', self.IP, self.PORT, ']')

    def socketClose(self):
        self.sock.close()
        print('UDP Socket Closed[', self.IP, self.PORT, ']')
        
    def recvfrom(self, data_size):
        return self.sock.recvfrom(data_size)
    
    def sendto(self, msg, address):
        self.sock.sendto(msg, address)
    
###--CarDate(UDP)--##############################

class VehicleInfoReceiver:
    def __init__(self, ip, port):
        self.IP = ip
        self.PORT = port
        self.ADDRESS = (ip, port)
        self.udp = UDP_Socket(ip, port, 'server')
        self.sock = self.udp.get_sock()
        self.LABELS = [ 'MsgID: ','Roll: ' ,'Pitch: ','Yaw: ','Vx: ','Vy: ','Vz: ','RollVel: ','PitchVel: ',
                       'YawVel: ','Ax: ','Ay: ','Az: ','RollAcc: ','PitchAcc: ','YawAcc: ','SteerAng: ',
                       'AccPedal: ','BrakePedal: ','GearNum: ','S_Vx: ','S_Vy: ','S_Vz: ','S_Roll: ','S_Pitch: ',
                       'S_Yaw: ','S_Ax: ','S_Ay: ','S_Az: ','S_RollAcc: ','S_PitchAcc: ','S_YawAcc: ',
                       'FL_rotv: ','FR_rotv: ','RL_rotv: ','RR_rotv: ','FL_rz: ','FR_rz: ','RL_rz: ','RR_rz: ', 
                       'Speed:' ]
        
    def get_socket(self):
        return self.sock
        
    def receive_datas(self):
        try:
            buffer, sender = self.udp.recvfrom(32)
            data_size = 1 + 8 * 39
            data, sender = self.udp.recvfrom(data_size)
            datas = struct.unpack('<Bddddddddddddddddddddddddddddddddddddddd', data)
            vx = datas[4]
            vy = datas[5]
            if vx < 0: vx = 0
            if vy < 0: vy = 0
            speed = (vx ** 2 + vy ** 2)**0.5 * 3.6
            datas += (speed,)
            return datas                   
        except Exception as e:
            print(e)
            return -1
        
    def make_print(self, datas):
        background = np.zeros((1000, 800), np.uint8)
        font = cv2.FONT_HERSHEY_PLAIN
        
        #receive_str = ''
        for idx, data in enumerate(datas):
            cv2.putText(background, self.LABELS[idx] + str(data), (10, 20*idx + 15), font, 1, (255,0,0), 2)
        return background
    
    def quit_receive(self):
        self.udp.socketClose()
        
class LidarDataReceiver:
    def __init__(self, ip, port):
        self.IP = ip
        self.PORT = port
        self.ADDRESS = (ip, port)
        self.udp = UDP_Socket(ip, port, 'server')
        self.sock = self.udp.get_sock()
        
    def get_socket(self):
        return self.sock
    
    def quit_receive(self):
        self.udp.socketClose()
        
class ControlSignalSender:
    def __init__(self, ip, port):
        self.IP = ip
        self.PORT = port
        self.ADDRESS = (ip, port)
        self.udp = UDP_Socket(ip, port, 'client')
        self.LABELS = ['MsgID: ', 'Accel: ', 'Brake: ', 'Gear: ', 'Steering Angle: ']
        
    def sendSignal(self, accel, brake, gear, stearing):
        msg = struct.pack('<Bdidd', 255, stearing, gear, accel, brake)
        self.udp.sendto(msg, self.ADDRESS)
        
    def make_print(self, accel, brake, gear, stearing):
        background = np.zeros((160, 320), np.uint8)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(background, self.LABELS[0] + '255', (10, 20), font, 1, (255,0,0), 2)
        cv2.putText(background, self.LABELS[1] + str(accel), (10, 40), font, 1, (255,0,0), 2)
        cv2.putText(background, self.LABELS[2] + str(brake), (10, 60), font, 1, (255,0,0), 2)
        cv2.putText(background, self.LABELS[3] + str(gear), (10, 80), font, 1, (255,0,0), 2)
        cv2.putText(background, self.LABELS[4] + str(stearing), (10, 100), font, 1, (255,0,0), 2)
        return background
    
    def quit_receive(self):
        self.udp.socketClose()
        
#TEST CODE(RUN THIS CODE TO TEST TCP SERVER)
if __name__ == '__main__':
    IP = '127.0.0.1'
    CAMERA_PORT = 2210
    VEHICLE_INFO_PORT = 50301
    CONTROL_SIGNAL_PORT = 50302
    
    tcp_camera = CamerReceiver(IP, CAMERA_PORT)
    
    udp_vehicleInfo = VehicleInfoReceiver(IP, VEHICLE_INFO_PORT)
    
    udp_controlSignal = ControlSignalSender(IP, CONTROL_SIGNAL_PORT)
    
    read_socket_list = [tcp_camera.get_socket(), udp_vehicleInfo.get_socket()]
    
    while True:
        conn_read_socket_list, conn_write_socket_list, conn_except_socket_list = select.select(read_socket_list, [], [])
        
        for conn_read_socket in conn_read_socket_list:
            if conn_read_socket == tcp_camera.get_socket():
                #RECEIVE CAMERA DATA
                try:
                    src, fps = tcp_camera.receive_img()
                    cv2.imshow("Camera", src)
                except Exception as e:
                    print(e)
            if conn_read_socket == udp_vehicleInfo.get_socket():
                #RECEIVE Vehicle Info DATA
                try:
                    carData = udp_vehicleInfo.receive_datas()
                    p_carData = udp_vehicleInfo.make_print(carData)
                    cv2.imshow("CarData", p_carData)
                except Exception as e:  
                    print(e)
        
        #SEND CONTROL SIGNAL
        try:
            accel = 0.5
            brake = 0.0
            gear = -1
            stearing = -1
            udp_controlSignal.sendSignal(accel, brake, gear, stearing)
            monitor = udp_controlSignal.make_print(accel, brake, gear, stearing)
            cv2.imshow("ControlSignal", monitor)
        except Exception as e:
            print(e)
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    
    tcp_camera.quit_receive()
    udp_vehicleInfo.quit_receive()
    udp_controlSignal.quit_receive()