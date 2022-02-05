import sys,os,cv2,time,threading,queue,snap7,matplotlib
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from matplotlib import pyplot as plt
matplotlib.use("Qt5Agg")
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from threading import Lock


dic={}
file = open('file.pbtxt', "r")
for line in file:
    data=line.strip().split("=")
    a=data[0]
    b=data[1]
    dic[a]=b
# print(dic)
user_dic={}
admins=(dic['admin'].strip().split(','))
for i in admins:
    user_data=i.split(':')
    admin=user_data[0]
    password=user_data[1]
    user_dic[admin]=password
# print(user_dic)
camera_ipaddress=dic['camera_ipaddress']
camera_username=dic['camera_username']
camera_password=dic['camera_password']
PLC_ipaddress=dic['plc_ipaddress']
read_camera=int(dic['read_camera'])
# print(read_camera)
frame_path=dic['frames_store_path']
folder_name=dic['folder_name']
read_video=dic['read_video']
ckpt_head=dic['ckpt_head']
ckpt_goggle=dic['ckpt_goggle']
a1=int(dic['x'])
b1=int(dic['y'])
w=int(dic['width'])
a2=a1+w
h=int(dic['height'])
b2=b1+h
th_H=float(dic['threshold_head'])
th_G=float(dic['threshold_goggle'])
sw=int(dic['show_height'])
sh=int(dic['show_width'])
head_aligned_crop_width=int(dic['head_aligned_crop_width'])
head_aligned_crop_height=int(dic['head_aligned_crop_height'])
ud_limit=int(dic['undetection_limit'])
tuning_limit=int(dic['fluctuation_frames_tuning'])
update_delay=int(dic['time_delay_for_display_label_update_in_milliseconds'])/1000
db_number=int(dic['plc_db_number'])
delay_for_save=int(dic['delay_for_save_in_milliseconds'])/1000
true_count=int(dic['true_count'])
backhead_width=int(dic['head_width_to_identify_as_a_backhead'])
buzzer_delay=int(dic['buzzer_delay'])
hc_limit=int(dic['head_count_limit'])
plc_delay=int(dic["plc_communication_delay_in_milliseconds"])/1000
giht=int(dic["goggle_inside_head_tolerance"])
folder_count_to_show=int(dic["folder_count_to_show_in_graph"])
prediction_list=[]

CUSTOM_MODEL_FOLDER_NAME = dic['model_folder_name'] 
LABEL_MAP_NAME = dic['label_map_name']
CUSTOM_MODEL_HEAD_FOLDER_NAME = dic['model_head_folder_name'] 
CUSTOM_MODEL_GOGGLE_FOLDER_NAME = dic['model_goggle_folder_name'] 
delay=int(dic['delay_time_in_milliseconds'])/1000
close_flag = False

# print(os.path.join(CUSTOM_MODEL_NAME, 'ckpt-{}'.format(ckpt)))
show_live_flag = False
shown_flag = False
save_flag = False
all_closed_flag = False
current_posX=0
current_posY=0
display=False
plc_send_goggle_data=False
detection_started_flag=False
detection_closed_flag=False
prediction_closed_flag=False

lock=Lock()
frame_queue=queue.Queue()
head_queue=queue.Queue()
crop_img_queue=queue.Queue()
goggle_queue=queue.Queue()
# plc_queue=queue.Queue()


files_H = {
    'PIPELINE_CONFIG':os.path.join(CUSTOM_MODEL_FOLDER_NAME,CUSTOM_MODEL_HEAD_FOLDER_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(CUSTOM_MODEL_FOLDER_NAME,CUSTOM_MODEL_HEAD_FOLDER_NAME, LABEL_MAP_NAME),
    'CKPT' : os.path.join(CUSTOM_MODEL_FOLDER_NAME,CUSTOM_MODEL_HEAD_FOLDER_NAME, 'ckpt-{}'.format(ckpt_head))#.format(int(ckpt)))
}
files_G = {
    'PIPELINE_CONFIG':os.path.join(CUSTOM_MODEL_FOLDER_NAME,CUSTOM_MODEL_GOGGLE_FOLDER_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(CUSTOM_MODEL_FOLDER_NAME,CUSTOM_MODEL_GOGGLE_FOLDER_NAME, LABEL_MAP_NAME),
    'CKPT' : os.path.join(CUSTOM_MODEL_FOLDER_NAME,CUSTOM_MODEL_GOGGLE_FOLDER_NAME, 'ckpt-{}'.format(ckpt_goggle))#.format(int(ckpt)))
}


# Load pipeline config and build a detection model
configs_H = config_util.get_configs_from_pipeline_file(files_H['PIPELINE_CONFIG'])
detection_model_H = model_builder.build(model_config=configs_H['model'], is_training=False)

# Restore checkpoint
ckpt_H = tf.compat.v2.train.Checkpoint(model=detection_model_H)
ckpt_H.restore(files_H['CKPT']).expect_partial()

@tf.function
@tf.autograph.experimental.do_not_convert
def detect_fn_H(image_H):
    image_H, shapes_H = detection_model_H.preprocess(image_H)
    prediction_dict_H = detection_model_H.predict(image_H, shapes_H)
    detections_H = detection_model_H.postprocess(prediction_dict_H, shapes_H)
    return detections_H


# Load pipeline config and build a detection model
configs_G = config_util.get_configs_from_pipeline_file(files_G['PIPELINE_CONFIG'])
detection_model_G = model_builder.build(model_config=configs_G['model'], is_training=False)

# Restore checkpoint
ckpt_G = tf.compat.v2.train.Checkpoint(model=detection_model_G)
ckpt_G.restore(os.path.join(files_G['CKPT'])).expect_partial()

@tf.function
@tf.autograph.experimental.do_not_convert
def detect_fn_G(image_G):
    image_G, shapes_G = detection_model_G.preprocess(image_G)
    #print(shapes)
    prediction_dict_G = detection_model_G.predict(image_G, shapes_G)
    detections_G = detection_model_G.postprocess(prediction_dict_G, shapes_G)
    return detections_G
    
    
category_index_H = label_map_util.create_category_index_from_labelmap(files_H['LABELMAP'])
category_index_G = label_map_util.create_category_index_from_labelmap(files_G['LABELMAP'])

print("Connecting to PLC...")
while True:
    try:
        client=snap7.client.Client()
        client.connect(PLC_ipaddress,0,1,102)
        print(bool(client.get_connected))
        print("PLC connected")
        break
    except:
        print("PLC not connected")
        time.sleep(.5)
        # break
    
# print(bool(client.get_connected))

# cap = cv2.VideoCapture()


class VideoCapture:
    def __init__(self, name):
        while True:
            self.cap = cv2.VideoCapture(name)
            if self.cap.isOpened():
                print("video source connected")
                break
            else:
                print("Retrying to connect video source")
        self.flag=False
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

  # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            if self.flag:
                break
            # time.sleep(.1)
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        lock.acquire()
        self.flag = True
        lock.release()

#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# cap = cv2.VideoCapture('rtsp://admin:Rane@123@{}/'.format(list1[0]))

def detection():
    global all_closed_flag,detection_started_flag,detection_closed_flag
    x1,x2,y1,y2,Info_head,X1,X2,Y1,Y2=0,0,0,0,"",0,0,0,0
    if read_camera:
        cap = VideoCapture('rtsp://{}:{}@{}/'.format(camera_username,camera_password,camera_ipaddress))
    else:
        cap = VideoCapture(read_video)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
    lock.acquire()
    detection_started_flag=True
    lock.release()
    while True: 
        start=time.perf_counter()
        or_read_img = cap.read()
        # print("test")
        frame_queue.put(or_read_img)
        frame = or_read_img[ b1:b2, a1:a2 ]
        image_np = np.array(frame)

        input_tensor_H = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections_H = detect_fn_H(input_tensor_H)
        # print(detections_H)
        num_detections_H = int(detections_H.pop('num_detections'))
        detections_H = {key: value[0, :num_detections_H].numpy()
                    for key, value in detections_H.items()}
        detections_H['num_detections'] = num_detections_H

        # detection_classes should be ints.
        detections_H['detection_classes'] = detections_H['detection_classes'].astype(np.int64)

        label_id_offset = 1
        # image_np_with_detections = image_np.copy()

        coordinates_H = viz_utils.return_coordinates(
                            image_np,#_with_detections,
                            np.squeeze(detections_H['detection_boxes']),
                            np.squeeze(detections_H['detection_classes']+label_id_offset),
                            np.squeeze(detections_H['detection_scores']),
                            category_index_H,
                            use_normalized_coordinates=True,
                            line_thickness=1,
                            min_score_thresh=th_H)
        head_queue.put(coordinates_H)

        if not coordinates_H==[]:
            l=len(coordinates_H)
            if l>1:
                if coordinates_H[0][2]>coordinates_H[1][2]:
                    coordinates_H.pop(0)
            y1=int(coordinates_H[0][0])
            y2=int(coordinates_H[0][1])
            x1=int(coordinates_H[0][2])
            x2=int(coordinates_H[0][3])
            Info_head=coordinates_H[0][5]
            dx=x2-x1
            dy=y2-y1
            # print(dx,dy)
            X1=round(x1-((head_aligned_crop_width-dx)/2))
            Y1=round(y1-((head_aligned_crop_height-dy)/2))
            X2=round(x2+((head_aligned_crop_width-dx)/2))
            Y2=round(y2+((head_aligned_crop_height-dy)/2))
            # print( X1-X2,Y1-Y2)
            if X1<0:
                X2=X2-X1
                X1=0
            if Y1<0:
                Y2=Y2-Y1
                Y1=0
            if X2>w:
                X1=X1-(X2-w)
                X2=w
            if Y2>h:
                Y1=Y1-(Y2-h)
                Y2=h
        else:
            x1,x2,y1,y2,Info_head=0,0,0,0,""

        coordinates_crop_img=[X1,X2,Y1,Y2]
        crop_img_queue.put(coordinates_crop_img)
        if Info_head=="Head":

            detection_head_frame= frame[ Y1:Y2,X1:X2]
            # detection_head_frame=cv2.resize(detection_head_frame1,(head_aligned_crop_width,head_aligned_crop_height))
            image_np_G = np.array(detection_head_frame)
        
            input_tensor_G = tf.convert_to_tensor(np.expand_dims(image_np_G, 0), dtype=tf.float32)
            detections_G = detect_fn_G(input_tensor_G)

            num_detections_G = int(detections_G.pop('num_detections'))
            detections_G = {key: value[0, :num_detections_G].numpy()
                            for key, value in detections_G.items()}
            detections_G['num_detections'] = num_detections_G

            # detection_classes should be ints.
            detections_G['detection_classes'] = detections_G['detection_classes'].astype(np.int64)

            label_id_offset_G = 1
            # image_np_with_detections_G = image_np_G.copy()

            coordinates_G = viz_utils.return_coordinates(
                                    image_np_G,#_with_detections,
                                    np.squeeze(detections_G['detection_boxes']),
                                    np.squeeze(detections_G['detection_classes']+label_id_offset_G),
                                    np.squeeze(detections_G['detection_scores']),
                                    category_index_G,
                                    use_normalized_coordinates=True,
                                    line_thickness=1,
                                    min_score_thresh=th_G)
            goggle_queue.put(coordinates_G)
            # print(coordinates_G)
        time.sleep(delay)
        finish=time.perf_counter()
        if all_closed_flag:
            cap.release()
            lock.acquire()
            detection_closed_flag=True
            lock.release()
            break
        print('{}ms'.format(round((finish-start)*1000)))
        # print("size",len(li),"    True:",li.count(True),"    False:",li.count(False))
        # count=count+.1

def prediction():
    global show_live_flag,prediction_list,save_flag,plc_send_goggle_data,detection_closed_flag,prediction_closed_flag
    x1_H,x2_H,y1_H,y2_H,percent_H,Info_H,crop_img_x1,crop_img_x2,crop_img_y1,crop_img_y2=0,0,0,0,0,"",0,0,0,0
    x1_G,x2_G,y1_G,y2_G,Info_G,status_G=0,0,0,0,"",""
    count_H,count_G,color,count_goggle_undetection=0,0,0,0
    prev_color=0
    hc=0
    count=0
    while True:
        or_img=frame_queue.get()
        predic_frame = or_img[ b1:b2, a1:a2 ]
        coor_H=head_queue.get()
        coor_crop_img=crop_img_queue.get()
        crop_img_x1=coor_crop_img[0]
        crop_img_x2=coor_crop_img[1]
        crop_img_y1=coor_crop_img[2]
        crop_img_y2=coor_crop_img[3]
        detection_head_frame= predic_frame[ crop_img_y1:crop_img_y2,crop_img_x1:crop_img_x2]
        # detection_head_frame= predic_frame[ 0:0,0:0]
        if not coor_H==[]:
            coor_G=goggle_queue.get()
            l=len(coor_H)
            if l>1:
                if coor_H[0][2]>coor_H[1][2]:
                    coor_H.pop(0)
            y1_H=int(coor_H[0][0])
            y2_H=int(coor_H[0][1])
            x1_H=int(coor_H[0][2])
            x2_H=int(coor_H[0][3])
            percent_H=round(int(coor_H[0][4]))
            Info_H=coor_H[0][5]
            if not coor_G==[]:
                y1_G=int(coor_G[0][0])
                y2_G=int(coor_G[0][1])
                x1_G=int(coor_G[0][2])
                x2_G=int(coor_G[0][3])
                percent_G=round(int(coor_G[0][4]))
                Info_G=coor_G[0][5]
            else:
                x1_G,x2_G,y1_G,y2_G,Info_G,status_G=0,0,0,0,"",""
        else:
            # x1_H,x2_H,y1_H,y2_H,percent_H,Info_H=0,0,0,0,0,""
            x1_H,x2_H,y1_H,y2_H,percent_H,Info_H=0,0,0,0,0,""

 
        
            # color=0
        # # print(coor_H,coor_G)
        if Info_H=="Head":
            hc=0
            if Info_G=="":
                if count_goggle_undetection<=ud_limit:
                    count_goggle_undetection=count_goggle_undetection+1
                # prediction_list.append(False)
                if count_goggle_undetection>ud_limit:
                    color=0
                    count_G=0
                    lock.acquire()
                    # plc_queue.put(True)
                    plc_send_goggle_data=True
                    lock.release()

            if Info_G=="Goggle":
                if ((x1_H-crop_img_x1-giht)<x1_G<((x1_H-crop_img_x1)+(x2_H-x1_H)-giht)) and ((y1_H-crop_img_y1-giht)<y1_G<((y1_H-crop_img_y1)+(y2_H-y1_H)-giht)) and ((x1_H-crop_img_x1+giht)<x2_G<((x1_H-crop_img_x1)+(x2_H-x1_H)+giht)) and ((y1_H-crop_img_y1+giht)<y2_G<((y1_H-crop_img_y1)+(y2_H-y1_H)+giht)):
                                    status_G="inside"
                                    if count_G<=tuning_limit:
                                        count_G=count_G+1                                   
                                    # count_goggle_undetection=0
                else:
                    status_G="outside"
                if count_G>tuning_limit:
                    color=255
                    count_goggle_undetection=0
                    lock.acquire()
                    # plc_queue.put(False)
                    plc_send_goggle_data=False
                    lock.release()
                    
                # prediction_list.append(True)
            # print("count_goggle_undetection:",count_goggle_undetection,"count_G:",count_G)

            
        #         # print("undetection")s

            
        else:
            if hc<=hc_limit:   
                hc=hc+1
            if hc>hc_limit:
                count_goggle_undetection=0
                count_G=0
                # color=255
                lock.acquire()
                # plc_queue.put(False)
                plc_send_goggle_data=False
                lock.release()
                # hc=0
        print("count_goggle_undetection:",count_goggle_undetection,"count_G:",count_G,"hc:",hc)
        if save_flag:
            if not (detection_head_frame.size==0):
                folder_path = os.path.join(frame_path,folder_name)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                date=time.strftime("%d-%m-%Y")
                frame_write_path=os.path.join(folder_path,date)
                if not os.path.exists(frame_write_path):
                    os.mkdir(frame_write_path)
                # if color==0:
                # write_path=os.path.join(frame_write_path,"NoGoggle")
                # if not os.path.exists(write_path):
                #     os.mkdir(write_path)
                export_path = os.path.join(frame_write_path,'NG{}.jpg'.format(time.strftime("%I.%M.%S_%p")))
                cv2.imwrite(export_path,detection_head_frame)

                # if color==255:
                #     write_path=os.path.join(frame_write_path,"Goggle")
                #     if not os.path.exists(write_path):
                #         os.mkdir(write_path)
                #     export_path = os.path.join(write_path,'G{}.jpg'.format(time.strftime("%I.%M.%S_%p")))
                #     cv2.imwrite(export_path,detection_head_frame)
                #     export_path = os.path.join(write_path,'G_rec{}.jpg'.format(time.strftime("%I.%M.%S_%p")))
                #     cv2.rectangle(detection_head_frame, (x1_G,y1_G), (x2_G, y2_G), (0,255,0), 2)
                #     cv2.putText(detection_head_frame, '{}:'.format(Info_G)+str(percent_G)+"%", (x1_G, y1_G-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2,cv2.LINE_8)
                #     cv2.imwrite(export_path,detection_head_frame)
                lock.acquire()
                save_flag = False
                lock.release()
        #     if Info_G=="Goggle":
        #         write_path=os.path.join(frame_write_path,"Goggle")
        #         if not os.path.exists(write_path):
        #             os.mkdir(write_path)
        #         export_path = os.path.join(write_path,'G{}.jpg'.format(time.strftime("%I.%M.%S_%p")))
        #         cv2.imwrite(export_path,detection_head_frame)
        #         export_path = os.path.join(write_path,'G_rec{}.jpg'.format(time.strftime("%I.%M.%S_%p")))
        #         cv2.rectangle(detection_head_frame, (x1_G,y1_G), (x2_G, y2_G), (0,255,0), 2)
        #         cv2.imwrite(export_path,detection_head_frame)
        # client.db_write(db_number,0,b'\x00')
                # count=count+1
                # print("saved",count)

        if show_live_flag:
            show_loop = True
            cv2.rectangle(or_img, (a1,b1), (a2, b2), (255,0,0), 6)
            cv2.putText(or_img, 'Detection_frame', (a1, b1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2,cv2.LINE_8)
            # cv2.rectangle(or_read_img, (x,y), (x+w, y+h), (255,0,0), 6)
            if Info_H=="Head":
                cv2.rectangle(predic_frame, (x1_H,y1_H), (x2_H, y2_H), (0,color,255), 6)
                cv2.putText(predic_frame, '{}:'.format(Info_H)+str(percent_H)+"%", (x1_H, y1_H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,color,255), 2,cv2.LINE_8)
                # prev_color=color
                if Info_G=="Goggle":
                    if status_G=="inside":
                        cv2.rectangle(predic_frame, (crop_img_x1+x1_G,crop_img_y1+y1_G), (crop_img_x1+x2_G, crop_img_y1+y2_G), (0,255,0), 6)
                        cv2.putText(predic_frame, '{}:'.format(Info_G)+str(percent_G)+"%", (crop_img_x1+x1_G, crop_img_y1+y1_G-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2,cv2.LINE_8)
                    if status_G=="outside":
                        cv2.rectangle(predic_frame, (crop_img_x1+x1_G,crop_img_y1+y1_G), (crop_img_x1+x2_G, crop_img_y1+y2_G), (0,0,255), 6)
                        cv2.putText(predic_frame, '{}:'.format(Info_G)+str(percent_G)+"%", (crop_img_x1+x1_G, crop_img_y1+y1_G-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2,cv2.LINE_8)  # else:
                        # count_G=count_G+1            
            cv2.imshow('object detection',cv2.resize(or_img,(sh,sw)) )
            if cv2.waitKey(1) & 0xFF == 27:
                lock.acquire()
                show_live_flag=False
                lock.release()

        if not (show_live_flag and show_loop):
                cv2.destroyAllWindows()
                # print("closed")
                lock.acquire()
                show_live_flag = False
                lock.release()
        show_loop = False
        if detection_closed_flag:
            cv2.destroyAllWindows()
            lock.acquire()
            prediction_closed_flag=True
            lock.release()
            break

def plc_communication():
    global prediction_list,save_flag,all_closed_flag,plc_send_goggle_data,prediction_closed_flag
    while True:
        plc_data=bytearray(client.db_read(db_number,0,1))
        snap7.util.set_bool(plc_data,0,0,True)
        if snap7.util.get_bool(plc_data,0,1):
            lock.acquire()
            save_flag=True
            lock.release()
            snap7.util.set_bool(plc_data,0,1,False)
        lock.acquire()
        state=plc_send_goggle_data
        lock.release()
        print(state)
        if state == True:
            snap7.util.set_bool(plc_data,0,2,True)
        if state == False:
            snap7.util.set_bool(plc_data,0,2,False)
        client.db_write(db_number,0,plc_data)
        if prediction_closed_flag:
            client.disconnect()
            break
        time.sleep(plc_delay)



t = threading.Thread(target=detection)
t.daemon = True
t.start()
z=0
while True:
    time.sleep(1)
    if detection_started_flag:
        print("detection started")
        break
    if z==1:
        print('waiting for detection start')
    if z<2:
        z=z+1

p = threading.Thread(target=prediction)
p.daemon = True
p.start()
time.sleep(1)
plc = threading.Thread(target=plc_communication)
plc.daemon = True
plc.start()
time.sleep(1)
# s = threading.Thread(target=save_frames)
# s.daemon = True
# s.start()
# time.sleep(1)

class firstWindow(QtWidgets.QMainWindow):
    def __init__(self):
        global current_posX,current_posY,shown_flag,movex_limit,movey_limit
        super(firstWindow, self).__init__()
        loadUi('page1.ui', self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        print("Ui_1")
        self.lineEdit_3.setPlaceholderText("User")
        self.lineEdit_4.setPlaceholderText("Password")
        icon = QtGui.QIcon('7.png')
        self.showPassAction = QtWidgets.QAction(icon, 'Show password', self)
        self.lineEdit_4.addAction(self.showPassAction, QtWidgets.QLineEdit.TrailingPosition)
        showPassButton = self.lineEdit_4.findChild(QtWidgets.QAbstractButton)
        showPassButton.pressed.connect(lambda: self.showPassword(True))
        showPassButton.released.connect(lambda: self.showPassword(False))

        # self.Indicator_label.setStyleSheet("border: 5px solid black; background-color: rgb(0, 255, 0);border-radius: 20px;")
        self.continueButton.clicked.connect(self.next)
        self.exitButton.clicked.connect(self.exit)
        if not shown_flag:
            self.show()
            current_posX=self.pos().x()
            current_posY=self.pos().y()
            shown_flag = True
        else:
            self.move(current_posX,current_posY)
            self.show()
        self.limit_x=movex_limit-self.geometry().width()
        self.limit_y=movey_limit-self.geometry().height()
        self.first_page_close=False
        # print(self.geometry().width(),self.geometry().height())
        # self.first_page_label_change_thread()

    def showPassword(self, show):
        self.lineEdit_4.setEchoMode(
            QtWidgets.QLineEdit.Normal if show else QtWidgets.QLineEdit.Password)

    def next(self):
        admin = self.lineEdit_3.text()
        password = self.lineEdit_4.text()
        w_offset=round((self.geometry().width()-151)/2)
        h_offset=round((self.geometry().height()-96)/2)
        print(bool(len(admin)))
        if bool(len(admin)):
            if bool(len(password)):
                # print(admin)
                # if admin.isdigit():
                    if admin in user_dic.keys():
                        if user_dic[admin]==password:
                            self.second=secondWindow(admin)
                            self.first_page_close=True
                            self.close()
                        else:
                            mbox = QMessageBox()  # popup the message box widget
                            mbox.setWindowTitle("Warning")
                            mbox.setText("Incorrect userID/password")
                            mbox.setIcon(QMessageBox.Warning)
                            mbox.move(self.x()+w_offset,self.y()+h_offset)
                            x = mbox.exec_()
                    else:
                        mbox = QMessageBox()  # popup the message box widget
                        mbox.setWindowTitle("Warning")
                        mbox.setText("You are not admin")
                        mbox.setIcon(QMessageBox.Warning)
                        mbox.move(self.x()+w_offset,self.y()+h_offset)
                        x = mbox.exec_()
                # else:
                #     mbox = QMessageBox()  # popup the message box widget
                #     mbox.setWindowTitle("Warning")
                #     mbox.setText("UserID must a number")
                #     mbox.setIcon(QMessageBox.Warning)
                #     mbox.move(self.x()+w_offset,self.y()+h_offset)
                #     x = mbox.exec_()
                    # mbox.width()
            else:
                mbox = QMessageBox()  # popup the message box widget
                mbox.setWindowTitle("Warning")
                mbox.width()
                mbox.setText("oops...  \nEnter a password")
                mbox.setIcon(QMessageBox.Warning)
                mbox.move(self.x()+w_offset,self.y()+h_offset)
                x = mbox.exec_()
        

        else:
            mbox = QMessageBox()  # popup the message box widget
            mbox.setWindowTitle("Warning")
            mbox.setText("oops...  \nEnter a userID")
            mbox.setIcon(QMessageBox.Warning)
            mbox.move(self.x()+w_offset,self.y()+h_offset)
            x = mbox.exec_()
            # print(mbox.width(),mbox.height(),mbox.exec_())


    def exit(self):
        global all_closed_flag
        # client.db_write(db_number,0,b'\x00')
        lock.acquire()
        all_closed_flag = True
        lock.release()
        t.join()
        p.join()
        plc.join()
        time.sleep(.5)
        app.quit()

    def mousePressEvent(self,event):
        self.oldPosition=event.globalPos()
        # print(self.x(),self.y())

    def mouseMoveEvent(self, event):
        global current_posX,current_posY
        delta= QPoint(event.globalPos() - self.oldPosition)
        # print(x,y,self.x(),self.y(),current_posX,current_posY)
        current_posX=self.x()+delta.x()
        current_posY=self.y()+delta.y()

        if (0 <= current_posX <= self.limit_x):
            #self.move(current_posX,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(current_posX,current_posY)
            elif (current_posY<0):
                self.move(current_posX,0)
            elif(current_posY>self.limit_y):
                self.move(current_posX,self.limit_y)
        elif (current_posX<0):
            #self.move(0,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(0,current_posY)
            elif (current_posY<0):
                self.move(0,0)
            elif(current_posY>self.limit_y):
                self.move(0,self.limit_y)
        elif(current_posX>self.limit_x):
            #self.move(self.limit_x,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(self.limit_x,current_posY)
            elif (current_posY<0):
                self.move(self.limit_x,0)
            elif(current_posY>self.limit_y):
                self.move(self.limit_x,self.limit_y)
        self.oldPosition=event.globalPos()

class secondWindow(QMainWindow):
    def __init__(self,admin):
        # genid=0
        global current_posX,current_posY,movex_limit,movey_limit,display
        super(secondWindow, self).__init__()
        loadUi('page2.ui', self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        print("Ui_2")
        self.live_detection_B.clicked.connect(self.startthread)
        self.violation_frames_B.clicked.connect(self.showrecords)
        self.violation_graph_B.clicked.connect(self.show_graph)
        
        self.logout_B.clicked.connect(self.logout)
        self.exit_B.clicked.connect(self.exit)
        self.welcome_L.setText("Welcome "+str(admin))
        self.move(current_posX,current_posY)
        self.show()
        self.limit_x=movex_limit-self.geometry().width()
        self.limit_y=movey_limit-self.geometry().height()
        self.second_page_close=False
        
        self.folder_path=os.path.join(frame_path,folder_name)
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        # self.second_page_label_change_thread()

    def startthread(self):
        global show_live_flag
        w_offset=round((self.geometry().width()-151)/2)
        h_offset=round((self.geometry().height()-96)/2)
        if not (show_live_flag):
            lock.acquire()
            show_live_flag = True
            lock.release()
            #self.statusLabel.setText("Live is running")
            #self.statusLabel.show()
        else:
            mbox = QMessageBox()  # popup the message box widget
            mbox.setWindowTitle("Warning")
            mbox.setText("Already running")
            mbox.setIcon(QMessageBox.Warning)
            mbox.move(self.x()+w_offset,self.y()+h_offset)
            x = mbox.exec_()
            print("Already running")

    def showrecords(self):
        # try:
            os.system('start {}'.format(self.folder_path))
        # except:
        #     mbox = QMessageBox()  # popup the message box widget
        #     mbox.setWindowTitle("Warning")
        #     mbox.setText("Path does not exist")
        #     mbox.setIcon(QMessageBox.Warning)
        #     x = mbox.exec_()

    def show_graph(self):
        totalFiles = 0
        totalDir = 0
        count=0
        dir_list=[]
        x=[]
        y=[]
        plt.close()
        self.fig= plt.figure()
        self.fig.patch.set_facecolor('gray')
        self.fig.patch.set_alpha(0.5)
        for base, dirs, files in os.walk(self.folder_path):
            for directories in dirs:
                dir_list.append(os.path.join(base,directories))
                totalDir += 1
            count=count+1
            dir_list.sort(key=os.path.getctime)
            for i in range(len(dir_list)):
                totalFiles=0
                # print("asd")
                # print(asdf[i])
                for B, D, F in os.walk(dir_list[i]):
                    # print(B)
                    # break
                    # print(F)
                    for a in F:
                # asd=os.path.join(base,Files)
                # print(date.fromtimestamp(os.stat(asd).st_mtime))
                        totalFiles += 1
                    break
                x.append(os.path.split(dir_list[i])[1])
                y.append(totalFiles)
            print(x)
            print(y)
            # plt.clf()
            plt.title("Violation Graph")
            plt.xlabel("Date")
            plt.ylabel("No of frames")
            plt.xlim(len(x)-folder_count_to_show-0.5,len(x))
            # for i in range(len(x)):
            #     plt.annotate(i,y[i],y[i],ha="center")
            # self.figure.patch.set_facecolor('blue')
            # self.figure.patch.set_alpha(0.5)
            plots=plt.bar(x,y)
            # plt.
            # plt.figure().na=
            # if count=

            for bar in plots:
                # print(bar.get_height())
                height=bar.get_height()
                plt.annotate('{}'.format(height),
                xy=(bar.get_x()+bar.get_width()/2,height),
                xytext=(0,2),
                textcoords='offset points',ha='center',va='center')

            # if count==1:
            break
        plt.show()
        # plt.figure().close()
        # print("asdfdfsds")


    def logout(self):
        global show_live_flag
        lock.acquire()
        self.second_page_close=True
        show_live_flag = False
        lock.release()
        print("Logged out")
        self.home = firstWindow()
        self.close()
        plt.close()

    def exit(self):
        global show_live_flag,all_closed_flag
        lock.acquire()
        show_live_flag = False
        all_closed_flag=True
        lock.release()
        t.join()
        p.join()
        plc.join()
        # client.db_write(db_number,0,b'\x00')
        time.sleep(.5)
        app.quit()

    def mousePressEvent(self,event):
        # print(event.globalPos())
        self.oldPosition=event.globalPos()

    def mouseMoveEvent(self, event):
        global current_posX,current_posY
        delta= QPoint(event.globalPos() - self.oldPosition)
        # print(x,y,self.x(),self.y(),current_posX,current_posY)
        current_posX=self.x()+delta.x()
        current_posY=self.y()+delta.y()

        if (0 <= current_posX <= self.limit_x):
            #self.move(current_posX,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(current_posX,current_posY)
            elif (current_posY<0):
                self.move(current_posX,0)
            elif(current_posY>self.limit_y):
                self.move(current_posX,self.limit_y)
        elif (current_posX<0):
            #self.move(0,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(0,current_posY)
            elif (current_posY<0):
                self.move(0,0)
            elif(current_posY>self.limit_y):
                self.move(0,self.limit_y)
        elif(current_posX>self.limit_x):
            #self.move(self.limit_x,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(self.limit_x,current_posY)
            elif (current_posY<0):
                self.move(self.limit_x,0)
            elif(current_posY>self.limit_y):
                self.move(self.limit_x,self.limit_y)
        self.oldPosition=event.globalPos()
# sw=secondWindow()


app = QApplication(sys.argv)
movex_limit=app.desktop().screenGeometry().width()
movey_limit=app.desktop().screenGeometry().height()
# print(QDesktopWidget().availableGeometry().width())
ex = firstWindow()
# second_page = secondWindow(123)
sys.exit(app.exec_())

# cap.release()
# cv2.destroyAllWindows()
