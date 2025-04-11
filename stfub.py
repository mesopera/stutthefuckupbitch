import cv2
import mediapipe as mp
import os
import sys
import platform
import threading
import pystray
from PIL import Image, ImageDraw

running = False
cap = None
tray_icon = None

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def is_middle_finger_up(landmarks):
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    fingers = [landmarks[tip].y < landmarks[pip].y for tip, pip in zip(tips, pip_joints)]
    return fingers == [False, True, False, False]

def shutdown():
    os_type = platform.system()
    if os_type == "Windows":
        os.system("shutdown /s /t 1")
    elif os_type in ("Linux", "Darwin"):
        os.system("shutdown now")

def detect_loop():
    global running, cap
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_middle_finger_up(hand_landmarks.landmark):
                    print("Middle finger detected! Shutting down...")
                    running = False
                    cap.release()
                    shutdown()
                    return
    if cap:
        cap.release()

def start_detection(icon, item=None):
    global running
    if not running:
        running = True
        threading.Thread(target=detect_loop, daemon=True).start()
        print("Detection started")

def stop_detection(icon, item=None):
    global running
    running = False
    print("Detection stopped")

def quit_program(icon, item=None):
    global running, cap
    running = False
    if cap:
        cap.release()
    icon.stop()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

def create_icon():
    icon_path = resource_path("icon.png")
    return Image.open(icon_path)

# Setup tray icon
icon_image = create_icon()
menu = pystray.Menu(
    pystray.MenuItem("Start Detection", start_detection),
    pystray.MenuItem("Stop Detection", stop_detection),
    pystray.MenuItem("Quit", quit_program)
)

tray_icon = pystray.Icon("MiddleFingerDetector", icon_image, "Middle Finger Detector", menu)
tray_icon.run()
