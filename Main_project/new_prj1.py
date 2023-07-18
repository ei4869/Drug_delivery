#小车1完成基础部分
import cv2
import numpy as np
from multiprocessing import Process, Queue, Event, Manager
from Faststdet import *
import serial
from collections import Counter
import time
from PID import *
from threading import Timer
import os,sys
import subprocess
WIDTH = 320
HEIGHT = 240

#串口指令
GO = "go000000\r\n"
STOP = "st000000\r\n"
BACK = "bc000000\r\n"
L90 = "tl000000\r\n"
R90 = "tr000000\r\n"
FWD = "fw000000\r\n"
RED = 'ar000000\r\n'
GREEN = 'cp000000\r\n'

#Roi参数

median_Roi = [50,190, 60, 280]
num_Roi1 = [120,240,50,280]
num_Roi2 = [100,220,10,350]

Kp1 = 0.3
Ki1 = 0.01
Kd1 = 0

# Kp2 = 0.38
# Ki2 = 0.013
# Kd2 = 0.01


Kp2 = 0.41
Ki2 = 0.0
Kd2 = 0.039
maxout = 40
minout = -40
maxI = 100
minI = -100
deadzone = 0

def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)

threshold = 125
previous_diff = None
def img_preprocess(img):
    global previous_diff
    # 将图像从BGR空间转换到HSV空间

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgred = hsv[median_Roi[0]:median_Roi[1],median_Roi[2]:median_Roi[3]]
    # 定义红色的阈值
    # 注意HSV空间的H通道表示色调，取值范围通常为0-180
    # 在OpenCV中，红色的H值通常在0-10和160-180这两个范围
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(imgred, lower_red, upper_red)

    lower_red = np.array([160, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(imgred, lower_red, upper_red)
    # 合并两个掩膜
    mask = cv2.bitwise_or(mask1, mask2)
    # 使用掩膜对图像进行二值化：红色区域为白色（255），非红色区域为黑色（0）
    Binary = cv2.bitwise_and(255*np.ones_like(mask), 255*np.ones_like(mask), mask=mask)
    median = cv2.bitwise_not(Binary)

    kernel = np.ones((3,3),np.uint8)
    median = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel,iterations=1)
    median = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel,iterations=1)
    cv2.imshow("median",median)

    # 选择扫描线的y坐标
    scanline_y =median.shape[0] // 2-30
    # 从左到右扫描
    for leftx in range(0, median.shape[1]):
        if median[scanline_y, leftx] == 0:
            break

    # 从右到左扫描
    for rightx in range(median.shape[1] - 1, -1, -1):
        if median[scanline_y, rightx] == 0:
            break

    # 计算黑线中心
    y = scanline_y+median_Roi[0]
    x = (leftx + rightx) // 2+median_Roi[2]
    center1 = (x,y)

    h,w = img.shape[:2]
    center2 = (w//2,h//2)

    #判断交叉路口
    delta = rightx - leftx

    # if previous_diff is None:
        # previous_diff = abs(center1[0] - center2[0])
    # else:
        # current_diff = abs(center1[0] - center2[0])
        # # 如果差值的变化过大，就让center1的x等于center2的x
        # if abs(current_diff - previous_diff) > 35:   # SOME_VALUE是你设定的阈值
           # center1 = (center2[0], center1[1])
        # # 更新previous_diff为当前的差值
        # previous_diff = current_diff

    #判断病房
    exitdata = np.sum(median==0)

    cv2.circle(img,center1,0,(0,0,255),10)
    cv2.circle(img,center2,0,(255,0,0),10)
    cv2.imshow("img",img)

    return center1,center2,delta,exitdata

#识别交叉路口
def iscross(delta):
    if delta > 100:
        return True
    else:
        return False

#识别出口
def isexit(exitdata):
    if exitdata > 5:
        return False
    else:
        return True

#鱼眼效果
def fish_eye(img, degree=0.15):
    height, width = img.shape[:2]
    map_x, map_y = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
    r = np.sqrt(map_x**2 + map_y**2)
    theta = np.arctan2(map_y, map_x)
    radius = r**(1-degree)
    map_x = radius * np.cos(theta)
    map_y = radius * np.sin(theta)
    map_x = (map_x + 1) * width / 2
    map_y = (map_y + 1) * height / 2
    map_x = np.clip(map_x, 0, width-1).astype('float32')
    map_y = np.clip(map_y, 0, height-1).astype('float32')
    unwrapped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return unwrapped_img

#线程处理：定时执行函数
def send_go(q3):
    q3.put(('Task', 'go'))
def DoneTurn(isDoneturn):
    isDoneturn.value = True
    print("DoneTurn:完全通过交叉路口")

degree = 0.15
QUEUE_MAX_SIZE = 20  # 设置队列的最大长度
def process1(q1, q2,isend):
    # 图像预处理进程
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    cnt = 0
    t1 = time.time()
    while True:
        cnt += 1
        ret, frame = capture.read()

        preprocessed_frame = img_preprocess(frame)
        #cv2.putText(frame,"FPS {0}".format("%.1f" % (cnt/(time.time()-t1))),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        #cv2.imshow("img1",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if q1.empty():
            q1.put(frame)
        # if not q1.full():
            # q1.put(frame)
        # if not q2.full():
            # q2.put(preprocessed_frame)
        if q2.empty():
            q2.put(preprocessed_frame)
        if isend.value:
            break
    cv2.destroyAllWindows()

def setflag(flag):
	flag = True
def process2( q2, q3, crossarray, isreturn, isDoneturn, isclose, isMED,isend,e,isreco,e2):
    global Kp1,Ki1,Kd1,Kp2,Ki2,Kd2,maxout,minout,maxI,minI,deadzone
    # PID寻线进程
    pid = CascadePIDController(Kp1,Ki1,Kd1,Kp2,Ki2,Kd2)
    flag1 = True
    flag2 = False
    while True:
        if e2.is_set():
            preprocessed_frame = q2.get()
            # time.sleep(0.01)
            #preprocessed_frame = q2.get()
            center1,center2,delta,exitdata = preprocessed_frame
            
            correction = pid.compute(center2,center1)
            q3.put(('pid', correction))
            
            if ((1 in crossarray) or (2 in crossarray)) and (not isreturn.value) and isDoneturn.value: 
                
                if isexit(exitdata):   #保证必须转弯完后开始判断
                    isreturn.value = True
                    q3.put(('Task','stop'))
                    q3.put(('Led','red'))
					
                    temp = True
                    while temp:
                        if isMED.value == False:
                            q3.put(('Task','back'))    #发送180°转弯指令
                            
                            temp = False
                        time.sleep(0.1)
                    t2 = Timer(5,lambda:q3.put(('Task','start')))   #定时发送go                       
                    t2.start()
            if  len(crossarray)==0 and isreturn.value:  #回到起点，停车
                if isexit(exitdata):
                    q3.put(('Task', 'stop'))
                    q3.put(('Led','green'))
                    
                    isend.value = True
                    e2.clear()
                    print("chongqi")
                    #restart_program()
                    os.execv(sys.executable, ['python'] + sys.argv)  # 重新启动当前python程序
                    # time.sleep(8)
                    # e.set()
                    # isreco.value = False

            if not isreturn.value:  #前往病房途中
                if iscross(delta) and isDoneturn.value: #识别到交叉路口且未转弯
                    if not isclose.value:       #不是近端交叉路口
                        print("到达交叉路口，开始识别数字")
                        q3.put(('Task', 'stop'))
                        q3.put(('recognition', 'on')) 
                        isDoneturn.value = False
                        e2.clear()  #关闭PID寻线进程   
                        #pid.clear()
                    else:
                        q3.put(('Task', 'stop'))
                        print("到达近端药房交叉路口")
                        isDoneturn.value = False
                        print("jiaocha",crossarray)
                        if len(crossarray) > 0:         #开始时识别到数字1或2，则近端路口转弯                
                            q3.put(('cross', crossarray[-1]))
                        else:           #没有识别到1或2，则近端路口直行
                            crossarray.append(0)
                            q3.put(('cross', crossarray[-1]))
                            isclose.value = False

            if isreturn.value:
                #print("cross:",delta)
                if iscross(delta) and isDoneturn.value:
                    if len(crossarray) > 0:
                        q3.put(('Task', 'stop'))
                        time.sleep(0.1)
                        direction = crossarray.pop(-1)
                        q3.put(('cross', direction))
                        isDoneturn.value = False
        if isend.value:
            break
def process3(q1, q3, e, crossarray,isclose, isMED,isreco,isend):
    global degree
    # 图像识别进程   
    recognitions = []
    detector = FastestDet(confThreshold=0.5,drawOutput=True)
    while True:
        if e.is_set():
            frame = q1.get()
            time.sleep(0.1)
            frame = q1.get()
            #frame = fish_eye(frame,degree)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #自适应阈值化
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            #将单通道二值图像转换为三通道图像
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if not isreco.value:    #初始为True，开启识别
                frame1 = frame[num_Roi1[0]:num_Roi1[1],num_Roi1[2]:num_Roi1[3]] #裁剪出识别数字区域
                result = detector.detect(frame1)
                #cv2.imshow("f1",frame1)
                if len(result) == 1:
                    recognitions.append(result[0][1])
                if len(recognitions) == 3:
                    FirstResults.append(max(set(recognitions), key=recognitions.count))
                    recognitions = []
                    if len(FirstResults)==1:
                        print("第一个数字识别完成,数字为：", FirstResults[0])
                    if len(FirstResults)==2:
                        print("第二个数字识别完成,数字为：",FirstResults[1])
                        q3.put(('recognition', 'off'))
                        temp = True
                        while temp:
                            if isMED.value == True:
                                print("启动PID")
                                q3.put(('Task','start'))
                                e2.set()       #识别第一个数字完成，启动PID 
                                isreco.value = True
                                if FirstResults[0] == '1':
                                    print("yes")
                                    isclose.value = True
                                    crossarray.append(1)    #近端路口左转
                                elif FirstResults[0] == '2':
                                    isclose.value = True
                                    crossarray.append(2)    #近端路口右转
                                temp = False
                            time.sleep(0.1)   
            elif isreco.value:   
                #print("recog:",recognitions)
                frame2 = frame[num_Roi2[0]:num_Roi2[1],num_Roi2[2]:num_Roi2[3]]
                result = detector.detect(frame2)
                for i in range(len(result)):
                    original_position = (result[i][0][0] + num_Roi2[2], result[i][0][1] + num_Roi2[0])
                    result[i] = (original_position, result[i][1], result[i][2])
                cv2.imshow("f2",frame2)
                if len(result)>0:
                    recognitions.append(result)
                if len(recognitions) == 10:     #10组识别得到最终识别结果
                        # 统计识别结果中出现次数最多的一组结果
                        counter = Counter(tuple(sorted(result)) for result in recognitions)
                        most_common_result = counter.most_common(1)[0][0]
                        # 获取出现次数最多的结果中的所有识别结果
                        target_results = [result for result in recognitions if tuple(sorted(result)) == most_common_result]
                        # 在target_results中查找序号与target_id相同的识别结果，并返回位置
                        center2 = (160,120)
                        target_position = center2
                        print("target:",target_results)
                        for result in target_results[0]:
                            target_digit = FirstResults[0]
                            if result[1] == target_digit:
                                FirstResults.remove(target_digit)
                                target_position = result[0]
                                print("交叉路口数字中含有目标数字",result[1])
                                break

                        delta = target_position[0]-center2[0]
                        print("转弯的delta:",delta)
                        q3.put(('recognition', 'off'))
                        
                        if delta > 0:
                            #右转
                            crossarray.append(2)
                        elif delta < 0:
                            #左转
                            crossarray.append(1)
                        else:
                            #直行
                            crossarray.append(0)
                        q3.put(('cross', crossarray[-1]))
                        t4 = Timer(2,lambda:e2.set())
                        t4.start()
                        #e2.set()       #识别数字完成，启动PID
                        recognitions = []
            #cv2.imshow('1',frame)
            if cv2.waitKey(1) == ord("q"):
                break
        if isend.value:
            break
    cv2.destroyAllWindows()

def process4(q3, e, isreturn,isend):
    # 串口通信进程
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    while True:
        try:
            result_type, result = q3.get()
            if result_type == 'pid':
                if result>=0:
                    data = "{:06d}".format(int(result))
                    ser.write(("rr"+data+"\r\n").encode('utf-8'))
                else:
                    result = -result
                    data = "{:06d}".format(int(result))
                    ser.write(("ll"+data+"\r\n").encode('utf-8'))
            elif result_type == 'Task':
                if result == 'stop':
                    ser.write(STOP.encode('utf-8'))
                elif result == 'back':
                    ser.write(BACK.encode('utf-8'))
                    print("返回")
                elif result == 'start':
                    ser.write(GO.encode('utf-8'))
            elif result_type == 'recognition':
                if result == 'on':
                    e.set()     #设置事件
                elif result == 'off':     
                    e.clear()   #清除事件
            elif result_type == 'cross':
                #一旦接收到cross信号，就开始计时，4s后发送转弯完成信号
                if isreturn.value:
                    t1 = Timer(3,DoneTurn,args=(isDoneturn,))
                elif not isreturn.value:
                    t1 = Timer(3,DoneTurn,args=(isDoneturn,))
                if result == 0:
                    ser.write(GO.encode('utf-8'))
                    print("直行")

                    # cnt = 0
                    # while time.time()-cnt<5:
                        # correction = 60
                        
                        # data = "{:06d}".format(int(correction))
                    
                        # ser.write(("ll"+data+"\r\n").encode('utf-8'))
                    #q3.put(('pid', correction))
                    
                    ser.write(GO.encode('utf-8'))
                elif result == 1:
                    if isreturn.value:
                        ser.write(R90.encode('utf-8'))
                        print("R90")
                    else:
                        ser.write(L90.encode('utf-8'))
                        print("L90")
                elif result == 2:
                    if isreturn.value:
                        ser.write(L90.encode('utf-8'))
                        print("L90")
                    else:
                        ser.write(R90.encode('utf-8'))
                        print("R90")
                t1.start()
                time.sleep(0.5)
            elif result_type == 'Led':
                if result == 'red':
                    ser.write(RED.encode('utf-8'))
                    print('红灯亮')
                elif result == 'green':
                    ser.write(GREEN.encode('utf-8'))
                    #print('绿灯亮')
            if isend.value:
                break
        except Exception:
            print("Serial Error")

def process5(isDoneturn,isMED,isend):
    #串口接收进程
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    while True:
        recv = ser.readline().decode().strip()
        # if recv == "DoneTurn":
        #     isDoneturn.value = True
        #     print("DoneTurn:完全通过交叉路口")
        if recv == "MEDOK":
            isMED.value = True
            print("收到药品")
        elif recv == "MEDTAKE":
            isMED.value = False
            print("取走药品")
        elif recv == "car1_2":
            os.system('sudo reboot now')
        ser.flushInput()
        if isend.value:
            break
        time.sleep(0.1)

if __name__ == '__main__':
    ser = serial.Serial("/dev/ttyAMA0", 115200,timeout=1)
    while True:
        recv = ser.readline().decode().strip()
        print("收到的数据：", recv)
        # if recv == "DoneTurn":
        #     isDoneturn.value = True
        #     print("DoneTurn:完全通过交叉路口")
        if recv == "car1_1":
            print("启动car1_1")
            break
        elif recv == "car1_2":
            print("启动car1_2")
            subprocess.call(['python', '/home/pi/Desktop/Drug_delivery (1)/Main_project/project2_1.py'])
            ser.close()
            break
        elif recv == "car1_3":
            print("启动car1_3")
            subprocess.call(['python', '/home/pi/Desktop/Drug_delivery (1)/Main_project/project2_2.py'])
            ser.close()
            print("over")
            break
        
        time.sleep(0.1)
    
    q1 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储原始图像
    q2 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储预处理的图像
    q3 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储串口通信的数据
    e = Event()
    e.set()     #开始时启动事件

    e2 = Event()
    e2.clear()  #开始时关闭事件

    manager = Manager()
    crossArray = manager.list()             #存储路口信息
    Firstresult = manager.Value('s',"")     #存储第一次识别结果
    isreturn = manager.Value('b',False)        #存储是否返回
    isDoneturn = manager.Value('b',True)    #防止识别完成数字后开启pid后继续识别到交叉路口
    isMED = manager.Value('b',False)
    isclose = manager.Value('b',True)
    FirstResults = manager.list()
    isreco = manager.Value('b',False)
	
    isend = manager.Value('b',False)
    p1 = Process(target=process1, args=(q1, q2,isend,))
    p2 = Process(target=process2, args=(q2, q3, crossArray, isreturn, isDoneturn, isclose, isMED,isend,e,isreco,e2))
    p3 = Process(target=process3, args=(q1, q3, e, crossArray,isclose, isMED,isreco,isend,))
    p4 = Process(target=process4, args=(q3, e,isreturn,isend,))
    p5 = Process(target=process5, args=(isDoneturn,isMED,isend))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()

    # while True:
        # if isend.value:
            # p1.terminate()
            # p2.terminate()
            # p3.terminate()
            # p4.terminate()
            # p5.terminate()
            # p1.join()
            # p2.join()
            # p3.join()
            # p4.join()
            # p5.join()
