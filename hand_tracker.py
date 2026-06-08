import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime

mp_hands = mp.solutions.hands