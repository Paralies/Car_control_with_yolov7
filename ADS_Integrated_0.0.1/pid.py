import time

class PID:
    def __init__(self, kp, ki, kd):
        #PID Gain값
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.previousTime = 0
        self.lastError = 0
        self.cumError = 0
        #self.previousTIme = time.time()
 
    def computePID(self, error): 
        #현재 시간 가져오기
        currentTime = time.time()
        #이전 분기와의 시간차 구하기(미분 위함)
        if self.previousTime != 0:
            elapsedTime = currentTime - self.previousTime
        else:
            self.previousTIme = time.time()
            elapsedTime = 0
        #적분 계산(오차값 x 시간)
        self.cumError += error * elapsedTime
        #미분 계산(기울기 계산)
        if elapsedTime != 0:
            rateError = (error - self.lastError) / elapsedTime
        else:
            rateError = 0
        #PID 합산
        out = self.kp * error + self.ki * self.cumError + self.kd * rateError
        #오차값 저장
        self.lastError = error
        #현재 분기 시간 저장
        self.previousTime = currentTime
        #계산값 리턴
        return out
    
#test code
if __name__ == '__main__':
    ex_pid = PID(1, 1, 1)
    print(ex_pid.computePID(10))