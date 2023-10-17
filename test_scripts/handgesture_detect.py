import cv2
import mediapipe as mp
import time

mp_draw=mp.solutions.drawing_utils
mp_hand=mp.solutions.hands
tipIds=[4,8,12,16,20]

def detect_hand_landmarks():
    """
    Detects hand landmarks using MediaPipe Hands.
    """
    with mp_hand.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9) as hands:
        cap = cv2.VideoCapture(0)
        while True:
            ret, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lmList = []
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    myHands = results.multi_hand_landmarks[0]
                    for id, lm in enumerate(myHands.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
            fingers = []
            if len(lmList) != 0:
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                finger_counter = fingers.count(1)

                if finger_counter == 2:
                    time.sleep(0.5)
                    print("Victory sign detected!")
                    return True
            else:
                print("No hand landmarks detected")
                continue
        cap.release()
        return False

if __name__ == "__main__":
    detect_hand_landmarks()

 