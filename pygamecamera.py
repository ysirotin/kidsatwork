import pygame
import pygame.camera
import dlib
import cv2
import numpy as np
import multiprocessing
import time
import threading

from pygame.locals import *

print('Importing face_recognition')
import face_recognition

DEVICE = '/dev/video0'
SIZE = (640, 480)

DISPLAY_SCALE = 2
DISPLAY_SIZE = (SIZE[0]*DISPLAY_SCALE, SIZE[1]*DISPLAY_SCALE)

SCALE = 0.25

#N_WORKERS = cpu_count()-1

# load images
moustache = cv2.resize(cv2.imread('moustache.png', cv2.IMREAD_UNCHANGED),(220,74))
moustache = cv2.copyMakeBorder( moustache, 83, 83, 10, 10, cv2.BORDER_REPLICATE)

groucho = cv2.resize(cv2.imread('groucho.png', cv2.IMREAD_UNCHANGED),(512,512))
groucho = cv2.copyMakeBorder( groucho, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

trunk = cv2.resize(cv2.imread('trunk.png', cv2.IMREAD_UNCHANGED),(100,152))
trunk = cv2.copyMakeBorder( trunk, 120, 10, 66, 96, cv2.BORDER_REPLICATE)

# image processing functions
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

def rgb_to_bgr(rgb):
    return rgb[...,::-1]

def bgr_to_rgb(bgr):
    return bgr[...,::-1]

def place_moustache(frame, points, moustache):
    # put a moustache onto the face
    # center midway between 33 & 51
    # rotation based on 27 and 30 angle
    # size based on distance between 0 and 16
    dx = (points['right_eye'][1][0] + points['right_eye'][4][0] - points['left_eye'][1][0] - points['left_eye'][4][0]) / 2
    dy = (points['right_eye'][1][1] + points['right_eye'][4][1] - points['left_eye'][1][1] - points['left_eye'][4][1]) / 2
    rot = -np.arctan2(dy,dx)/np.pi*180

    dx = points['chin'][0][0] - points['chin'][16][0]
    dy = points['chin'][0][1] - points['chin'][16][1]
    
    mw0 = moustache.shape[1]
    mw = (dx**2 + dy**2)**0.5

    cx = int((points['nose_tip'][2][0] + points['top_lip'][3][0])/2)
    cy = int((points['nose_tip'][2][1] + points['top_lip'][3][1])/2)    
   
    msz = cv2.resize(moustache, (0,0), fx=mw/mw0, fy=mw/mw0)
    mrot = imrotate(msz, rot)
    overlay(frame,mrot,cx,cy)

def place_groucho(frame, points, groucho):
    # put a groucho onto the face
    # center midway between 33 & 51
    # rotation based on 27 and 30 angle
    # size based on distance between 27 and 30
    dx = (points['right_eye'][1][0] + points['right_eye'][4][0] - points['left_eye'][1][0] - points['left_eye'][4][0]) / 2
    dy = (points['right_eye'][1][1] + points['right_eye'][4][1] - points['left_eye'][1][1] - points['left_eye'][4][1]) / 2
    rot = -np.arctan2(dy,dx)/np.pi*180

    dx = points['right_eye'][1][0] - points['left_eye'][4][0]
    dy = points['right_eye'][1][1] - points['left_eye'][4][1]
    
    mh0 = groucho.shape[0]
    mh = 3 * (dx**2 + dy**2)**0.5

    cx = int((points['nose_bridge'][0][0] + points['nose_bridge'][3][0]) / 2)
    cy = int((points['nose_bridge'][0][1] + points['nose_bridge'][3][1]) / 2)
    
    gsz = cv2.resize(groucho, (0,0), fx=mh/mh0, fy=mh/mh0)
    grot = imrotate(gsz, rot)
    overlay(frame,grot,cx,cy)

def draw_points(frame, points, scale=1):
    for feature in points.keys():
        for pt in points[feature]:
            cv2.circle(frame,(int(pt[0] / scale), int(pt[1] / scale)), 2, color=[0,255,0], thickness=-1)

def draw_text(frame, pt, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    margin = 6
    (w, h) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]

    # set the text start position
    x, y = pt

    # make the coords of the box with a small padding of two pixels
    box_coords = ((x, y), (x + w - 2 + 2*margin, y - h - 2 - 2*margin))

    cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, text, (x+margin, y-margin), font, fontScale=font_scale, color=(255, 255, 255), thickness=2)

def draw_face_rect(frame, rect, text=None):
    cv2.rectangle(frame,
                  (rect[1],rect[0]),
                  (rect[3],rect[2]),
                  color=[0,255,0],
                  thickness=2,
                  lineType=cv2.LINE_AA)
    if text:
        draw_text(frame, (rect[3],rect[2]), text)

# get the standard face image
def get_face(frame, rect):
    x0, y1, x1, y0 = rect

    pad = int(0.5 * (x1-x0))
    frame_pad = cv2.copyMakeBorder(frame, top=pad, bottom=pad, left=pad, right=pad,
                                   borderType= cv2.BORDER_CONSTANT, value=[0, 0, 0])
    x1 += pad * 2
    y1 += pad * 2
    face = frame_pad[x0:x1, y0:y1, :].copy()
    face = cv2.resize(face, (100, 100))
    return face

# function to identify a template
def identify_template(template, threshold=0.4):
    ret = None
    if templates_ref:
        scores = face_recognition.face_distance(templates_ref, template)
        min_score = min(scores)
        ind_min = scores.tolist().index(min_score)
        if min_score < threshold:
            ret = ind_min
    return ret


# function to enroll faces in frame
def enroll_faces(frame, rects):

    # TO DO: save face image clip to display after identification (standard size)
    
    frame_temp = frame.copy()
    for rect in scaled_rects:
        draw_face_rect(frame_temp, rect, 'enrolling')
    draw_frame(frame_temp)
    
    templates = face_recognition.face_encodings(frame, known_face_locations=scaled_rects)
    for template, rect in zip(templates, rects):
        ind = identify_template(template)
        if ind is None:
            person_id = 'person%002d' % len(frames_ref)
            draw_text(frame, (20,SIZE[1]-20), person_id)
            frames_ref.append(get_face(frame,rect))
            templates_ref.append(template)
            person_ref.append(person_id)
        else:
            person_id = person_ref[ind]

        draw_face_rect(frame, rect, person_id)
        draw_frame(frame)
        pygame.time.wait(2000)

# function to identify faces in frame
def identify_faces(frame, rects):

    # TO DO: display gallery image (in lower corner)
    
    names = []
    inds = []
    templates = face_recognition.face_encodings(frame, known_face_locations=rects)
    for template in templates:
        ind = identify_template(template)
        if ind is not None:
            names.append(person_ref[ind])
            inds.append(ind)
        else:
            names.append('???')
            inds.append(None)

    return names, inds

def get_frame():
    return cv2.transpose(cv2.flip(rgb_to_bgr(pygame.surfarray.array3d(screen)), 0))

def draw_frame(frame):
    # draw frame onto the screen and then onto the display
    pygame.surfarray.blit_array(screen, bgr_to_rgb(cv2.transpose(frame)))
    pygame.transform.scale(screen, DISPLAY_SIZE, display)
    pygame.display.flip()

def process_frame(frame):
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray_full, (0,0), fx=SCALE, fy=SCALE)
        
    rects = face_recognition.face_locations(gray)
    points = face_recognition.face_landmarks(gray,face_locations=rects)
    scaled_rects = [tuple([int(ind / SCALE) for ind in rect]) for rect in rects]
    scaled_points = [{key: [tuple([int(value / SCALE) for value in point]) for point in group[key]] for key in group} for group in points]
    return scaled_rects, scaled_points


if __name__ == '__main__':

    print('Initializing pygame...')
    pygame.init()
    
    display = pygame.display.set_mode(DISPLAY_SIZE, pygame.FULLSCREEN)
    screen = pygame.surface.Surface(SIZE, 0)

    templates_ref = []
    frames_ref = []
    person_ref = []

    pygame.camera.init()
    camera = pygame.camera.Camera(DEVICE, SIZE, 'MJPG')
    camera.start()
    
    POINTS = False
    GROUCHO = False
    MOUSTACHE = False
    CAPTURE = True

    while(CAPTURE):

        camera.get_image(screen)
        frame = get_frame()                

        scaled_rects, scaled_points = process_frame(frame)
        
        for pts in scaled_points:
            try:
                if POINTS:
                    draw_points(frame, pts)
                elif MOUSTACHE:
                    place_moustache(frame, pts, moustache)
                elif GROUCHO:
                    place_groucho(frame, pts, groucho)
            except Exception as e:
                print(e)
                pass
                
        draw_frame(frame)
        
        # respond to keyboard
        for event in pygame.event.get():
            if (event.type == QUIT) or (event.type == KEYDOWN and event.key == K_q):
                CAPTURE = False
            elif event.type == KEYDOWN and event.key == K_s:
                pygame.image.save(screen, FILENAME)
            elif event.type == KEYDOWN and event.key == K_p:
                POINTS = not POINTS
                GROUCHO = False
                MOUSTACHE = False
            elif event.type == KEYDOWN and event.key == K_g:
                GROUCHO = not GROUCHO
                POINTS = False
                MOUSTACHE = False
            elif event.type == KEYDOWN and event.key == K_m:
                MOUSTACHE = not MOUSTACHE
                POINTS = False
                GROUCHO = False
            elif event.type == KEYDOWN and event.key == K_e:
                enroll_faces(frame, scaled_rects)                
            elif event.type == KEYDOWN and event.key == K_i:
                frame_temp = frame.copy()
                for rect in scaled_rects:
                    draw_face_rect(frame_temp, rect, 'identifying')
                draw_frame(frame_temp)   
                names, inds = identify_faces(frame, scaled_rects)
                for rect, name, ind in zip(scaled_rects, names, inds):
                    draw_face_rect(frame, rect, name)
                    if ind is not None:
                        h,w,_ = frame.shape
                        frame[h-101:h-1,w-101:w-1,:] = frames_ref[ind]
                        cv2.rectangle(frame, (w-1,h-1), (w-101,h-101),
                                      color=[0,255,0],
                                      thickness=2,
                                      lineType=cv2.LINE_AA)
                draw_frame(frame)
                pygame.time.wait(2000)

    camera.stop()
    pygame.quit()
    
