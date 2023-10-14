import cv2
import mediapipe as mp

def detect_thumbs_up():
    cap = cv2.VideoCapture(0)  # Open the default camera (change index if using a different camera)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    while True:
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate the landmarks for thumb, index, and middle fingers
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Check if the thumb tip is above the index and middle finger tips
                if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
                    # Print the gesture in the terminal
                    print("Thumbs up detected!")

        #cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_thumbs_up()
