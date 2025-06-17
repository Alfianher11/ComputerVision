import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Inisialisasi kamera dan canvas
cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

def is_index_finger_up(hand_landmarks):
    # Titik telunjuk: 8 (ujung), 6 (bawah)
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y

def is_middle_finger_up(hand_landmarks):
    return hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Koordinat ujung telunjuk
            x_tip = int(hand_landmarks.landmark[8].x * w)
            y_tip = int(hand_landmarks.landmark[8].y * h)

            index_up = is_index_finger_up(hand_landmarks)
            middle_up = is_middle_finger_up(hand_landmarks)

            # Mode menggambar: hanya telunjuk yang terangkat
            if index_up and not middle_up:
                cv2.circle(frame, (x_tip, y_tip), 10, (0, 255, 0), -1)
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x_tip, y_tip
                cv2.line(canvas, (prev_x, prev_y), (x_tip, y_tip), (0, 255, 0), 5)
                prev_x, prev_y = x_tip, y_tip

            # Jika telunjuk & jari tengah terangkat, berhenti menggambar
            elif index_up and middle_up:
                prev_x, prev_y = 0, 0

    # Gabungkan frame dan canvas
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    draw_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame = cv2.add(frame_bg, draw_fg)

    cv2.putText(frame, "Tekan 'c' untuk hapus canvas", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Virtual Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
