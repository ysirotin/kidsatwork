# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:17:15 2019

@author: ysirotin
"""

import cv2
import dlib
import time
import numpy as np
import subprocess as sub
import pandas as pd
import os
import uuid
import glob
from threading import Thread, Event

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

moustache = cv2.resize(cv2.imread('moustache.png', cv2.IMREAD_UNCHANGED),(220,74))
moustache = cv2.copyMakeBorder( moustache, 83, 83, 10, 10, cv2.BORDER_REPLICATE)

groucho = cv2.resize(cv2.imread('groucho.png', cv2.IMREAD_UNCHANGED),(512,512))
groucho = cv2.copyMakeBorder( groucho, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

trunk = cv2.resize(cv2.imread('trunk.png', cv2.IMREAD_UNCHANGED),(100,152))
trunk = cv2.copyMakeBorder( trunk, 120, 10, 66, 96, cv2.BORDER_REPLICATE)

def imrotate(img,angle):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv2.warpAffine(img, M, (cols,rows))

def overlay(img1,img2,x_o,y_o):
    h = img2.shape[0]
    w = img2.shape[1]
    
    y1, y2 = y_o - int(h/2), y_o + (h-int(h/2))
    x1, x2 = x_o - int(w/2), x_o + (w-int(w/2))

    alpha2 = img2[:, :, 3] / 255.0
    alpha1 = 1.0 - alpha2
    
    for c in range(0, 3):
        img1[y1:y2, x1:x2, c] = (alpha2 * img2[:, :, c] +
                                  alpha1 * img1[y1:y2, x1:x2, c])

def place_moustache(frame,ptsx,ptsy,moustache):
    # put a moustache onto the face
    # center midway between 33 & 51
    # rotation based on 27 and 30 angle
    # size based on distance between 0 and 16
    dx = (ptsx[46]+ptsx[43])/2 - (ptsx[37]+ptsx[40])/2
    dy = (ptsy[46]+ptsy[43])/2 - (ptsy[37]+ptsy[40])/2
    rot = -np.arctan2(dy,dx)/np.pi*180
    
    dx = ptsx[0] - ptsx[16]
    dy = ptsy[0] - ptsy[16]
    
    mw0 = moustache.shape[1]
    mw = 0.75 * (dx**2 + dy**2)**0.5
    
    cx = int((ptsx[33] + ptsx[51]) / 2)
    cy = int((ptsy[33] + ptsy[51]) / 2)
   
    msz = cv2.resize(moustache, (0,0), fx=mw/mw0, fy=mw/mw0)
    mrot = imrotate(msz, rot)
    
    overlay(frame,mrot,cx,cy)

def place_trunk(frame,ptsx,ptsy,trunk):
    # put a trunk onto the face
    # center midway between 33 & 51
    # rotation based on 27 and 30 angle
    # size based on distance between 27 and 30
    dx = ptsx[27] - ptsx[30]
    dy = ptsy[27] - ptsy[30]
    rot = -90-np.arctan2(dy,dx)/np.pi*180
        
    mh0 = trunk.shape[0]
    mh = 7.5 * (dx**2 + dy**2)**0.5
    
    cx = int((ptsx[27] + ptsx[30]) / 2)
    cy = int((ptsy[27] + ptsy[30]) / 2)
    
    tsz = cv2.resize(trunk, (0,0), fx=mh/mh0, fy=mh/mh0)
    trot = imrotate(tsz, rot)
    
    overlay(frame,trot,cx,cy)

def place_groucho(frame,ptsx,ptsy,groucho):
    # put a groucho onto the face
    # center midway between 33 & 51
    # rotation based on 27 and 30 angle
    # size based on distance between 27 and 30
    dx = (ptsx[46]+ptsx[43])/2 - (ptsx[37]+ptsx[40])/2
    dy = (ptsy[46]+ptsy[43])/2 - (ptsy[37]+ptsy[40])/2
    rot = -np.arctan2(dy,dx)/np.pi*180
    
    dx = ptsx[43] - ptsx[40]
    dy = ptsy[43] - ptsy[40]
    
    mh0 = groucho.shape[0]
    mh = 3 * (dx**2 + dy**2)**0.5
    
    cx = int((ptsx[27] + ptsx[30]) / 2)
    cy = int((ptsy[27] + ptsy[30]) / 2)
    
    gsz = cv2.resize(groucho, (0,0), fx=mh/mh0, fy=mh/mh0)
    grot = imrotate(gsz, rot)
    
    overlay(frame,grot,cx,cy)



#%% openbr stuff
min_score = 0.8
br_loc = r'C:\Program Files\OpenBR\bin\br.exe'

def br_enroll(img_file):
    print('Enrolling: ' + img_file)
    br_command = r'%s -algorithm FaceRecognition -enroll %s gallery.gal' % (br_loc,img_file)
    print(br_command)
    sub.call(br_command)
    
def br_compare(img_file):
    br_command = r'"%s" -algorithm FaceRecognition -compare gallery.gal %s scores.csv' % (br_loc,img_file)
    sub.call(br_command)

    name = 'Unrecognized'
    path = None
    try:
        data = pd.read_csv('scores.csv')
        names = [os.path.basename(f).split('.')[0] for f in data.columns[1:]]
        n_match = np.zeros(len(names))        
        for ii in range(len(data)):
            scores = data.loc[ii,data.columns[1:]].values
            p_max = np.argmax(scores)
            v_max = scores[p_max]
            print(v_max)
            if v_max>min_score: # only count a match if above criterion
                n_match[p_max]+=1
                
        if (np.max(n_match)>0):
            p_max = np.argmax(n_match)
            name = names[p_max]
            path = data.columns[p_max+1]
    except:
        pass
    
    return name,path
            

MOUSTACHE = False
POINTS = False
GROUCHO = False
ELEPHANT = False

# enroll all images in gallery
img_files = glob.glob(os.path.sep.join(['.','gallery','*.jpg']))
for img_file in img_files:
    br_enroll(img_file)

class FrameGrabber(Thread):
    def __init__(self, source_id):
        # set up video
        self.cap = cv2.VideoCapture(source_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        ret, self.cur_frame = self.cap.read()
        self.stop = Event()
        self.fps = 0
        Thread.__init__(self)
        print('frame grabber initialized')
    
    def finish(self):
        self.cap.release()
        print('frame grabber finished')
    
    def run(self):
        print('frame grabber is running')
        t0 = time.time()
        while not self.stop.isSet():
            try:
                if self.cap.grab():
                    ret, self.cur_frame = self.cap.retrieve()
                    time.sleep(0.0666)
                    tf = time.time()
                    self.fps = 1/(tf-t0)
                    t0 = tf
            except:
                self.stop.set()
        
        self.finish()
            
myGrabber = FrameGrabber(0)
myGrabber.start()

SCALE = 0.25
while(True):
    frame = myGrabber.cur_frame.copy()
    print(myGrabber.fps)
    # Our operations on the frame come here
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0,0), fx=SCALE, fy=SCALE)
    rects = detector(gray, 1)
    for rect in rects:
        pts = predictor(gray, rect)
        ptsx = [int(pts.part(ii).x / SCALE) for ii in range(pts.num_parts)]
        ptsy = [int(pts.part(ii).y / SCALE) for ii in range(pts.num_parts)]
        if POINTS:
            for ii in range(0,pts.num_parts):
                cv2.circle(frame,(ptsx[ii],ptsy[ii]),2,color=[0,255,0],thickness=-1)        
        
        try:
            if MOUSTACHE:
                place_moustache(frame,ptsx,ptsy,moustache)
            elif GROUCHO:
                place_groucho(frame,ptsx,ptsy,groucho)
            elif ELEPHANT:
                place_trunk(frame,ptsx,ptsy,trunk)

        except Exception as e:
            print(e)
        
    # Display the resulting frame
    cv2.imshow('webcam',frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('m'):
        #toggle moustache
        MOUSTACHE = not MOUSTACHE
        if MOUSTACHE:
            GROUCHO = False
            ELEPHANT = False
    elif key & 0xFF == ord('g'):
        GROUCHO = not GROUCHO
        if GROUCHO:
            MOUSTACHE = False
            ELEPHANT = False
    elif key & 0xFF == ord('t'):
        ELEPHANT = not ELEPHANT
        if ELEPHANT:
            MOUSTACHE = False
            GROUCHO = False
    elif key & 0xFF == ord('p'):
        POINTS = not POINTS
    elif key & 0xFF == ord('e'):
        img_file = os.path.sep.join(['.','gallery',str(uuid.uuid4())+'.jpg'])
        cv2.imwrite(filename=img_file,img=frame)
        br_enroll(img_file)
        cv2.putText(frame, img_file, (10,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('webcam',frame)
        key = cv2.waitKey(1)
        time.sleep(1)
    elif key & 0xFF == ord('i'):
        try:
            img_file = os.path.sep.join(['.','probes',str(uuid.uuid4())+'.jpg'])
            cv2.imwrite(filename=img_file,img=frame)
            name, path = br_compare(img_file)
            if path:
                frame = cv2.imread(path)
                cv2.putText(frame, path, (10,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else:
                cv2.putText(frame, 'Unrecognized', (10,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        except:
            cv2.putText(frame, 'Error', (10,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('webcam',frame)
        key = cv2.waitKey(1)
        time.sleep(1)
        
# When everything done, release the capture
myGrabber.stop.set()
myGrabber.join()

cv2.destroyAllWindows()
