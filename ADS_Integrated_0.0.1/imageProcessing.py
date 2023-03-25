import numpy as np
import cv2
import math

class imgProcessing:
    def __init__(self):
        self.accel = 0
        self.brake = 1
        self.gear = 0
        self.stear = 0
        
    def hsv2binary(self, hsv):
        hsvLower = np.array([0, 0, 0])    # 추출할 색의 하한(HSV)
        hsvUpper = np.array([0, 120, 120])    # 추출할 색의 상한(HSV)
        hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSV에서 마스크를 작성
        return hsv_mask
    
    def region_of_interest(self, img, vertices):
            mask = np.zeros_like(img)

            if len(img.shape) > 2:
                channel_count = img.shape[2]
                ignore_mask_color = (255,) * channel_count
            else:
                ignore_mask_color = 255

            cv2.fillPoly(mask, vertices, ignore_mask_color)

            masked_image = cv2.bitwise_and(img, mask)
            return masked_image
    
    def weighted_img(self, img, initial_img, a=0.8, b=1.,c=0.):
            return cv2.addWeighted(initial_img, a, img, b, c)

    def draw_error(self, src, hsv, target_y, min_y):
        hsv = self.hsv2binary(hsv)
        h, w = hsv.shape
        background = np.copy(src)
        GREEN = [0, 255, 0]
        RED = [0, 0, 255]
        
        middle_x = int(w/2)
        #draw middle line
        cv2.line(background, (middle_x, target_y - 10), (middle_x, target_y + 10), GREEN, 2)
        
        left_x = middle_x
        right_x = middle_x
        #핸들
        while hsv[target_y, left_x] == 255:
            left_x -= 1
            if left_x == 0:
                break
        while hsv[target_y, right_x] == 255:
            right_x += 1
            if right_x == w:
                break
        #Draw Base Line
        cv2.line(background, (left_x, target_y), (right_x, target_y), GREEN, 2)
        
        target_middle = int((left_x + right_x) / 2)

        cv2.line(background, (target_middle, target_y-10), (target_middle, target_y+10), RED, 2)
        cv2.line(background, (left_x, target_y - 10), (left_x, target_y + 10), RED, 2)
        cv2.line(background, (right_x, target_y - 10), (right_x, target_y + 10), RED, 2)

        #엑셀
        dist_y = target_y
        while hsv[dist_y, int(w/2)] == 255:
            dist_y -= 1
            if dist_y == min_y:
                break
        cv2.line(background, (int(w/2) - 15, dist_y), (int(w/2) + 15, dist_y), [0,0,255], 2)
            
        #브레이크
        cv2.line(background, (int(w/2), dist_y), (int(w/2), min_y), [0,0,255], 2)
        
        accel_error = target_y - dist_y
        brake_error = dist_y - min_y
        handle_error = middle_x - target_middle
        
        return background, (accel_error, brake_error, handle_error)
    
    def draw_monitor(self, queue):
        monitor = background = np.zeros((240, 640), np.uint8)
        WHITE = [255, 255, 255]
        for i in range(1, 640):
            cv2.line(background, (i-1, int(120 - queue[i-1])), (i, int(120 - queue[i])), WHITE, 1)
        return monitor