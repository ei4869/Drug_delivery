#协同：小车1送药到远端病房
import cv2
import numpy as np
from multiprocessing import Process, Queue, Event, Manager
from Faststdet import *
import serial
from collections import Counter
import time
from PID import *
import re
from threading import Timer
import subprocess
WIDTH = 320
HEIGHT = 240

#给单片机串口指令
GO = "go000000\r\n"
STOP = "st000000\r\n"
BACK = "bc000000\r\n"
L90 = "tl000000\r\n"
R90 = "tr000000\r\n"
FWD = "fw000000\r\n"

#给小车2
go = "go\r\n"
ONGO = "ongo\r\n"
LEFT = "left\r\n"
RIGHT = "right\r\n"

L = "l\r\n"
R = "r\r\n"
#Roi参数
median_Roi = [50,190, 80, 260]

num_Roi1 = [120, 240, 160, 310]
num_Roi2 = [120, 240, 10, 160]
num_Roi = [120,240,50,280]
Kp1 = 0.3
Ki1 = 0.01
Kd1 = 0

Kp2 = 0.4
Ki2 = 0.0
Kd2 = 0.040

previous_diff = None
def fisheye(image):
    height, width, _ = image.shape
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            map_x[i, j] = width / 2 + (j - width / 2) * ((i / (height / 2)) ** 0.35)
            map_y[i, j] = i
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

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

    #开闭运算
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
    delta = rightx - leftx

    if previous_diff is None:
        previous_diff = abs(center1[0] - center2[0])
    else:
        current_diff = abs(center1[0] - center2[0])
        # 如果差值的变化过大，就让center1的x等于center2的x
        if abs(current_diff - previous_diff) > 35:   # SOME_VALUE是你设定的阈值
           center1 = (center2[0], center1[1])
        # 更新previous_diff为当前的差值
        previous_diff = current_diff

    exitdata = np.sum(median==0)

    cv2.circle(img,center1,0,(0,0,255),10)
    cv2.circle(img,center2,0,(255,0,0),10)
    cv2.imshow("img",img)
    return center1,center2,delta,exitdata

def iscross(delta):
    if delta > 130:
        return True
    else:
        return False

#识别出口
def isexit(exitdata): 
    if exitdata > 5:
        return False
    else:
        return True

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
def send_go():
    q3.put(('Task', 'go'))

def DoneTurn(isDoneturn):
    isDoneturn.value = True
    print("DoneTurn:完全通过交叉路口")
    
QUEUE_MAX_SIZE = 10  # 设置队列的最大长度

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
        if q2.empty():
            q2.put(preprocessed_frame)
        if isend.value:
            break    
    cv2.destroyAllWindows()
def process2(q2, q3, crossarray, isreturn, isDoneturn, isclose, isMED,is2stop,isend):
    global Kp1,Ki1,Kd1,Kp2,Ki2,Kd2,maxout,minout,maxI,minI,deadzone
    # PID寻线进程
    pid = CascadePIDController(Kp1,Ki1,Kd1,Kp2,Ki2,Kd2)
  
    while True:
        if e2.is_set():
            preprocessed_frame = q2.get()
            center1,center2,delta,exitdata = preprocessed_frame
            correction = pid.compute(center2,center1)        
            q3.put(('pid', correction))

            if ((1 in crossarray) or (2 in crossarray)) and (not isreturn.value) and isDoneturn.value: 
                if isexit(exitdata):   #保证必须转弯完后开始判断
                    isreturn.value = True
                    q3.put(('Task','stop'))
                    temp = True
                    while temp:
                        if not isMED.value:
                            q3.put(('Task','back'))    #发送180°转弯指令
                            q4.put(('MSGto2','go'))   #2车启动
                            temp = False
                        time.sleep(0.1)
                    t1 = Timer(3,lambda:q3.put(('Task','start')))   #定时发送go
                    t1.start()
            if  len(crossarray)==0 and isreturn.value:  #回到起点，停车
                if True:
                    q3.put(('Task', 'stop'))
                    
                    isend.value = True
                    e2.clear()
                    ser = serial.Serial("/dev/ttyAMA3", 115200)
                    ser.close()
                    
                    subprocess.call(['python', '/home/pi/Desktop/Drug_delivery (1)/Main_project/new_prj1.py'])
                    
            if not isreturn.value:
                if iscross(delta) and isDoneturn.value:
                    if not isclose.value:
                        print("到达交叉路口，开始识别数字")
                        q3.put(('Task', 'stop'))
                        q3.put(('recognition', 'on')) 
                        isDoneturn.value = False
                        e2.clear()  #关闭PID寻线进程   
                    else:
                        q3.put(('Task', 'stop'))
                        print("到达近端药房交叉路口")
                        isDoneturn.value = False
                        if len(crossarray) > 0:         #开始时识别到数字1或2，则近端路口转弯                
                            q3.put(('cross', crossarray[-1]))
                        else:           #没有识别到1或2，则近端路口直行
                            crossarray.append(0)
                            q3.put(('cross', crossarray[-1]))
                            isclose.value = False

            if isreturn.value:
                if iscross(delta) and isDoneturn.value:
                    if len(crossarray) > 0:
                        q3.put(('Task', 'stop'))
                        time.sleep(0.1)
                        if len(crossarray) == 3:
                            q3.put(('Task', 'stop'))
                            print("停车等待小车2到达暂停点")
                            choseDir = crossarray[-1]
                            if choseDir == 1:
                                choseDir = 2
                            elif choseDir == 2:
                                choseDir = 1
                            choseDir = str(choseDir)
                            q4.put(('MSGto2', choseDir))    #单字符
                            temp2 = True
                            while temp2:
                                if is2stop.value:                                   
                                    temp2 = False
                                time.sleep(0.1)
                            t2 = Timer(3, lambda: q4.put(('MAGto2', 'ongo')))   
                            #车2停车等待后，车1预留3s转弯时间，完全转过后发信息给车2
                            t2.start()
                        direction = crossarray.pop(-1)
                        q3.put(('cross', direction))
                        isDoneturn.value = False
        if isend.value:
            break
                #计时3s保证车转过弯后，让2车直走无视交叉路口直到到达病房或者继续前进到病房（根据前面的远端路口方向信息比较）  

def process3(q1, q3, e, crossarray, isMED,isreco,isend):
    global degree
    # 图像识别进程   
    recognitions = []
    detector = FastestDet(drawOutput=True)
    while True:
        if e.is_set():
            frame = q1.get()
            #frame = fish_eye(frame,degree)
            #frame = fisheye(frame)
           
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #自适应阈值化
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            # 将单通道二值图像转换为三通道图像
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if not isreco.value:
                frame = frame[num_Roi[0]:num_Roi[1],num_Roi[2]:num_Roi[3]]
                result = detector.detect(frame)
                #cv2.imshow('1',frame)
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
                        data = 'num'+FirstResults[1]
                        q4.put(('MSGto2', data))        #发送第二个数字给小车2
                        temp = True
                        while temp:
                            if isMED.value == True:
                                print("启动PID")
                                q3.put(('Task','start'))
                                e2.set()       #识别第一个数字完成，启动PID 
                                isreco.value = True
                                temp = False
                            time.sleep(0.1)   
            else:                     
                result = detector.detect(frame)
                if len(result)>0:
                    recognitions.append(result)
                if len(recognitions) == 3:     #10组识别得到最终识别结果
                        # 统计识别结果中出现次数最多的一组结果
                        counter = Counter(tuple(sorted(result)) for result in recognitions)
                        most_common_result = counter.most_common(1)[0][0]

                        # 获取出现次数最多的结果中的所有识别结果
                        target_results = [result for result in recognitions if tuple(sorted(result)) == most_common_result]

                        # 在target_results中查找序号与target_id相同的识别结果，并返回位置
                        center2 = (160,120)
                        target_position = center2
                        target_position2 = center2
                        for result in target_results[0]:
                            target_digit = FirstResults[0]
                            
                            terget_digit2 = FirstResults[1]
                            if result[1] == target_digit2:
                                target_position2 = result[0]
                            if result[1] == target_digit:
                                target_position = result[0]
                                print("交叉路口数字中含有目标数字",result[1])
                                #break
                        delta2 = target_position2[0]-center2[0]      
                        delta = target_position[0]-center2[0]
                        q3.put(('recognition', 'off'))
                        e2.set()       #识别数字完成，启动PID
                        print("转弯的delta:",delta)
                        if delta > 0:
                            #右转
                            crossarray.append(2)
                        elif delta < 0:
                            #左转
                            crossarray.append(1)
                        else:
                            #直行
                            crossarray.append(0)
                        if delta2<0:
                            q4.put(('MSGto2', 'l'))
                        elif delta2>0:
                            q4.put(('MSGto2', 'r'))
                        q3.put(('cross', crossarray[-1]))
                        recognitions = []
            if cv2.waitKey(1) == ord("q"):
                break
            #time.sleep(0.05)
        if isend.value:
            break   
    cv2.destroyAllWindows()
    
def process4(q3, e, isreturn, isDoneturn,isend):
    # 串口通信进程
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    while True:   
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
            elif result == 'start':
                ser.write(GO.encode('utf-8'))
        elif result_type == 'recognition':
            if result == 'on':
                e.set()     #设置事件
            elif result == 'off':       
                e.clear()   #清除事件
        elif result_type == 'cross':
            if isreturn.value:
                t3 = Timer(3,DoneTurn,args=(isDoneturn))
            elif not isreturn.value:
                t3 = Timer(3,DoneTurn,args=(isDoneturn))
            if result == 0:
                ser.write(GO.encode('utf-8'))
                print("直行")
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
            t3.start()
        if isend.value:
            break
        # elif result_type == 'MSGto2':
            # if result == 'back':        #使小车2在暂停点掉头
                # ser.write(BACK.encode('utf-8'))
            # elif result == 'run':       #使小车2启动所有程序，从识别数字开始
                # ser.write(RUN.encode('utf-8'))
            # elif result == 'ongo':      #使小车2直行，忽略交叉路口直到到达病房，或者继续前进，2车需要对1车发的ongo信息根据病房方向进行判断
                # ser.write(ONGO.encode('utf-8'))
            # elif bool(re.match(r'^num\d+$',result)):      #1发送2的目标病房号
                # num = re.findall(r'\d+',result)[0]
                # data = "{:06d}".format(int(num))
                # ser.write(("dg"+data+"\r\n").encode('utf-8'))
            # elif result == 'go':        #使小车2启动，识别完数字后的启动
                # ser.write(GO.encode('utf-8'))
            # elif bool(re.match(r'^\d+$',result)):      #根据1的远端交叉路口转弯方向，确定2在其相反方向暂停（加定时，保证转过弯后并实现暂停后才发信息2STOP表示2已到暂停点），实现暂停后再比较方向信息并发2STOP信息给1车，此后2车等待1车回传ongo信息，判断继续前进还是掉头到病房
                # direct = int(result)        #2根据暂停的方向和原本该去的病房号进行比较，若相同则2不停车直接到目标病房，若不同，则2等待1通过路口后掉头直行
                # if direct == 1:                   
                    # ser.write(L90.encode('utf-8'))
                # elif direct == 2:
                    # ser.write(R90.encode('utf-8'))

def process5(isMED,is2stop,isend):
    #串口接收进程
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    ser2 = serial.Serial("/dev/ttyAMA3", 115200)
    while True: 
        recv = ser.readline().decode().strip()
        recv2 = ser.readline().decode().strip()
        if recv2 == "2STOP":
            is2stop.value = True
            print("小车2到达暂停点")
        if recv == "DoneTurn":
            isDoneturn.value = True
            print(recv)
        elif recv == "MEDOK":
            isMED.value = True
            print("收到药品")
        elif recv == "MEDTAKE":
            isMED.value = False
            print("取走药品")
        # elif recv == "2STOP":
            # is2stop.value = True
            # print("小车2到达暂停点")
        ser.flushInput()
        if isend.value:
            break
def process6(q4,isend):
    ser = serial.Serial("/dev/ttyAMA3", 115200)
    while True:   
        result_type, result = q4.get()
        if result_type == 'MSGto2':
            if result == 'go':        ##使小车2在暂停点掉头
                ser.write(go.encode('utf-8'))
            elif bool(re.match(r'^num\d+$',result)):      #1发送2的目标病房号
                num = re.findall(r'\d+',result)[0]
                data = "{:06d}".format(int(num))
                ser.write(("dg"+data+"\r\n").encode('utf-8'))
            elif result == "ongo":
                ser.write(ONGO.encode('utf-8'))
            elif result == "1":
                ser.write(LEFT.encode('utf-8'))
            elif result == "2":
                ser.write(RIGHT.encode('utf-8'))
            elif result == "l":
                ser.write(L.encode('utf-8'))
            elif result == "r":
                ser.write(R.encode('utf-8'))
        if isend.value:
            break
#单片机收到的指令优先级：Go>stop>back=fwd=r90=l90>l=r

if __name__ == '__main__':
    q1 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储原始图像
    q2 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储预处理的图像
    q3 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储串口通信的数据
    q4 = Queue(maxsize=QUEUE_MAX_SIZE)      #与车2通信的数据
    
    e = Event()
    e.set()     #开始时启动事件

    e2 = Event()
    e2.clear()  #开始时关闭事件

    manager = Manager()
    crossArray = manager.list()             #存储路口信息
    Firstresult = manager.Value('s',"")     #存储第一次识别结果
    isreturn = manager.Value('b',True)        #存储是否返回
    isDoneturn = manager.Value('b',True)    #防止识别完成数字后开启pid后继续识别到交叉路口
    isMED = manager.Value('b',False)
    isclose = manager.Value('b',True)
    isreco = manager.Value('b',False)
    FirstResults = manager.list()
    is2stop = manager.Value('b',False)
    
    isend = manager.Value('b',False)
    p1 = Process(target=process1, args=(q1, q2,isend,))
    p2 = Process(target=process2, args=(q2, q3, crossArray, isreturn, isDoneturn, isclose, isMED,is2stop,isend,))
    p3 = Process(target=process3, args=(q1, q3, e, crossArray, isMED,isreco,isend,))
    p4 = Process(target=process4, args=(q3, e,isreturn,isDoneturn,isend,))
    p5 = Process(target=process5, args=(isMED,is2stop,isend,))
    p6 = Process(target=process6, args=(q4,isend,))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
