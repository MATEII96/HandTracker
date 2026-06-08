import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


FINGER_JOINTS = {
    'Police': [1, 2, 3, 4],
    'Aratator': [5, 6, 7, 8],
    'Mijlociu': [9, 10 ,11 ,12],
    'Inelar': [13, 14, 15, 16],
    'Mic': [17, 18, 19, 20],  
}
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

HAND_MASS_KG = 0.45
PALM_REAL_M = 0.10

def hand_position_meters(lms, w, h):

    wx, wy = lms[0].x * w, lms[0].y * h
    mx, my = lms[9].x * w, lms[9].y * h
    palm_px = np.hypot(mx -wx, my-my)
    

               