import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import csv

from scipy.signal import argrelmin, argrelmax
from scipy import interpolate
from sympy import Symbol

from multiprocessing import Process
import time

import threading

import speech2text_20211102_new
import sound_cut
import short_speech

import pandas as pd

import pyaudio  #録音機能を使うためのライブラリ
import wave     #wavファイルを扱うためのライブラリ

count=0

name_list=['abe','asada','akiyoshi','ueda','okami','kanda','kodama','tainaka']
sentence_list=["sentence{}".format(i) for i in np.arange(1,3)]
count_list=["count{}".format(i) for i in np.arange(1,4)]
stroke_speed_list=["slow","fast"]

df=pd.read_csv('./csv/experiment_1109_new.csv',encoding='SHIFT-JIS')

#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap.set(cv2.CAP_PROP_FOCUS, 255)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

# 動画ファイル保存用の設定
#fps = int(cap.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 


fps=30
#fps=25 
#fps = 20

#w=640
#h=480            # カメラの縦幅を取得
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
video_list=list() 

for i in name_list:
    for ii in stroke_speed_list:
        for iii in count_list:
            for iiii in sentence_list:
                video_list.append("{}_{}_{}_{}".format(i,ii,iii,iiii))  

video=cv2.VideoWriter("movie/{}.mp4".format(video_list[count]), fourcc, fps, (w, h)) # 動画の仕様（ファイル名、fourcc, FPS, サイズ）


#基本情報の設定
#FORMAT = pyaudio.paInt16 #音声のフォーマット
#CHANNELS = 1             #モノラル
#RATE = 44100             #サンプルレート
    
#CHUNK = 2**11            #データ点数

def mediapipe_detection(frame):

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    results=hands.process(frame_rgb)
    ii=results.multi_hand_landmarks
    if ii:
        for i in ii:
            mpDraw.draw_landmarks(frame,i,mpHands.HAND_CONNECTIONS)

    gray_new=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    global threshold
    ret_,threshold=cv2.threshold(gray_new,0,255,cv2.THRESH_OTSU)

def get_center(img):

    try:
        y,x=np.where(img==255)
        x_avg,y_avg=np.average(x),np.average(y)

        return [int(x_avg),int(y_avg)]

    except:
        try:
            return [int(x_avg),int(y_avg)]
            #None
        
        except:
            None

def recording_sound(name,ind,all_start):
    
    
 
    #global stream,audio
    #start_voice,RATE,CHUNK,RECORD_SECONDS

    RECORD_SECONDS = 100 #録音する時間の長さ（秒）
    WAVE_OUTPUT_FILENAME = "./wav/{}.wav".format(name[ind]) #音声を保存するファイル名
    iDeviceIndex = 0 #録音デバイスのインデックス番号
 
    #基本情報の設定
    FORMAT = pyaudio.paInt16 #音声のフォーマット
    CHANNELS = 1             #モノラル
    RATE = 44100             #サンプルレート
    
    CHUNK = 2**11            #データ点数
    audio = pyaudio.PyAudio() #pyaudio.PyAudio()
 
    stream = audio.open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        input_device_index = iDeviceIndex, #録音デバイスのインデックス番号
        frames_per_buffer=CHUNK)


    
    start_video=time.time()
    print('video_{}'.format(start_video-all_start))

    time.sleep(10-(start_video-all_start))
    print('video_{}'.format(time.time()-all_start))


    #--------------録音開始---------------

    print ("recording...")
    frames = []
    try:
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            #print('--------------------------------')

       

    except :
        print ("finished recording")
        
    
        #--------------録音終了---------------
    
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

    

 
        

def recording_movie(video,count):

    #global count
    center_list=list()
    cnt__=0

    

    try:    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
        while True:
            ret, frame = cap.read()  

            #fps = int(cap.get(cv2.CAP_PROP_FPS))   
            #print(fps)

            mediapipe_detection(frame)    
            lower_filter,higher_filter=np.array([224,220,224]),np.array([224,227,224])
            #lower_filter,higher_filter=np.array([0,254,0]),np.array([0,255,0])
            
            inRange_mask=cv2.inRange(frame,lower_filter,higher_filter)
            bitwise=cv2.bitwise_and(frame,frame,mask=inRange_mask)
            bitwise_gray=cv2.cvtColor(bitwise,cv2.COLOR_BGR2GRAY)
            ret__,bitwise_threshold=cv2.threshold(bitwise_gray,0,255,cv2.THRESH_OTSU)
            center=get_center(bitwise_threshold)
            if center!=None:
                center_list.append(center)
                if len(center_list)==1:
                    speed_realtime=0
                
                else:
                    speed_realtime=abs(center_list[-1][0]-center_list[-2][0])

                if cnt__%6==0:
                    print('stroke_speed={}[cm/s]'.format((speed_realtime/(1/fps))/54))
                
                cv2.circle(bitwise,center,0,[150,255,30],15,200)
                

            video.write(bitwise) 
            
            #start_video=time.time()
            #print('video_{}'.format(start_video-all_start))

            cv2.imshow('frame',bitwise)
            key_=cv2.waitKey(1) 

            cnt__+=1

        
    except:
            
        global x,center_new_list

        center_new_list=[i[0] for i in center_list]
        x=np.arange(0,len(center_new_list)*(1/fps),1/fps)

        speed___avg=np.sum(abs(np.diff(center_new_list)))/((len(center_new_list)-1)*(1/fps))
            
        #speed___avg=np.sum(abs(np.diff([ko[0] for ko in center_list])))/((len(center_list)-1)*(1/fps))
        #-0.0001
        #with open("hand_{}_{}.csv".format(video_list[count],count),"w") as f:
           # writer=csv.writer(f)
        #for jkl in center_new_list:
            #writer.writerow(jkl)
        #print(center_new_list)

        global df

        inddd=list(df[df.iloc[:,0]==name_list[0]].index)
        df.iloc[inddd[count],4]=speed___avg/51
            
        if count<6:
            df.iloc[inddd[count],5]=50/(len(center_new_list)*(1/fps))
        else:
            df.iloc[inddd[count],5]=65/(len(center_new_list)*(1/fps))

        center_all_list.append(center_new_list)
        #print(center_all_list)
        center_index_l=[ind__ for ind__,kl in enumerate(center_new_list) if ind__%10 ==0]

        f_in = interpolate.Akima1DInterpolator(x[center_index_l], np.array(center_new_list)[center_index_l])
        f_in_center=f_in(x)

        max_index=argrelmax(np.array(f_in_center))
        min_index=argrelmin(np.array(f_in_center))

        #fig=plt.figure()

        plt.text(0,0,'speed_avg={}[cm/s]'.format(speed___avg/51))

        plt.plot(x,np.array(center_new_list)/51,label='center')
        plt.plot(x,np.array(f_in_center)/51,'y--', label='scipy')
        plt.plot(x[max_index[0]], np.array(center_new_list)[max_index[0]]/51,'ro')
        plt.plot(x[min_index[0]], np.array(center_new_list)[min_index[0]]/51, 'bo')

        

        max_list=[[iu,iuu] for iu,iuu in zip(x[max_index[0]],np.array(center_new_list)[max_index[0]])]
        min_list=[[iu_,iuu_] for iu_,iuu_ in zip(x[min_index[0]],np.array(center_new_list)[min_index[0]])]
        max_min_list=list()



        
        for iiij,[ij,kj] in enumerate(zip(max_list,min_list)):
            if ij[0]<kj[0]:
                max_min_list+=[ij[1],kj[1]]
            else:
                max_min_list+=[kj[1],ij[1]]
        if len(max_list)>len(min_list):
            max_min_list+=[max_list[-1][1]]

        elif len(max_list)<len(min_list):
            max_min_list+=[min_list[-1][1]]

        else:
            None


        print(np.array(max_min_list)/51)
        print(abs(np.diff(max_min_list)/51))


        df.iloc[inddd[count],6]=np.sum(abs(np.diff(max_min_list)/51))/len(max_min_list)

        df.iloc[inddd[count],7]=len(max_min_list)

        print(center_new_list)
        print(abs(np.diff(center_new_list)))
            
       
        #input_file="./kishi.wav"
        #.format(video_list[count])
        #sound_cut.main(input_file)
        #short_speech.transcribe_file(input_file)

        plt.grid()
        plt.xlabel('Time[sec]')
        plt.ylabel('center_placement')
        plt.title('center_placement_graph')
        plt.tight_layout()
        plt.legend()
        #plt.show()


        #fig.savefig("C:/Users/imdam/Desktop/opencv_exp/opencv_detection_center/data/{}.png".format(str(video_list[count])))
        fig.savefig("img/{}.png".format(video_list[count]))

        

        
                                # フレームを取得


if __name__ == '__main__':

    with open("csv/hand_{}_1113.csv".format(name_list[0]),"w",newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['---'+str(i_fghj)+'---' for i_fghj in np.arange(1,11) ])

        center_all_list=list()
        
    
        while True:

            ret,frame=cap.read()
            mediapipe_detection(frame)
            cv2.imshow('frame',frame)
            key=cv2.waitKey(1)
                
            if key==27:
                center_all_new_list=list()
                for iuy in center_all_list:
                    if len(iuy) < np.max([len(ser) for ser in center_all_list]):
                        center_all_new_list.append(iuy+list(np.zeros(np.max([len(tgb) for tgb in center_all_list])-len(iuy))))

                    else:
                        center_all_new_list.append(iuy)


                #print(center_all_list)
                writer.writerows(np.array(center_all_new_list).T)
                df.to_csv('./csv/experiment_1109_new.csv',encoding='SHIFT-JIS',index=False)

                break

            elif key==ord('r'):
                
                all_start=time.time()
        
                video=cv2.VideoWriter("movie/{}.mp4".format(video_list[count]), fourcc, fps, (w, h)) # 動画の仕様（ファイル名、fourcc, FPS, サイズ）
                
                fig=plt.figure()


                #p=Process(target=recording_sound, args=(video_list,count,all_start))
                p=Process(target=speech2text_20211102_new.main, args=[all_start])
                
                p.start()
                #time.sleep(5)
                

                #recording_sound(video_list,count,all_start)


                #with open("csv/hand_{}.csv".format(video_list[count]),"w") as f:
                    #writer=csv.writer(f)
                    #process1 = Process(name="process1", target=speech2text_20211102.main)
                    #process2 = Process(name="process2", target=recording_movie, args=(video, count))
                    #process2.start()
                    #process1.start()
                #thread_1 = threading.Thread(target=speech2text_20211102.main())
                #thread_2 = threading.Thread(target=recording_movie(video, count))
                
                #thread_1.start()
                #thread_2.start()

                
                print('start_movie={}'.format(time.time()-all_start))

                for jjjh in range(10):
                    print(10-jjjh)
                    time.sleep(1)

                print('start_movie_new={}'.format(time.time()-all_start))

                recording_movie(video,count)

                dfdf_cs=pd.read_csv('./csv/aswsa_.csv',encoding='SHIFT-JIS',header=None)
                idx_list=list()
                idx_list_y=list()

                for num in dfdf_cs.iloc[0,:]:
                    idx = np.abs(np.asarray(x) - float(num)).argmin()
                    idx_list.append(x[idx])
                    idx_list_y.append(center_new_list[idx])

                plt.scatter(idx_list,np.array(idx_list_y)/51)

                plt.show()
                #speech2text_20211102.main()
                count+=1

    
        



