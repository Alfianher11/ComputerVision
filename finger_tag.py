import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi Hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Buka webcam
cap = cv2.VideoCapture(0)

# ID titik ujung jari (thumb tip, index tip, middle tip, ring tip, pinky tip)
finger_tips = [4, 8, 12, 16, 20]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip horizontal dan konversi warna ke RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar koneksi dan titik tangan
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Menandai ujung jari
            for idx in finger_tips:
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[idx].x * w)
                y = int(hand_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"Tip {idx}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Tampilkan hasil
    cv2.imshow("Jari Terdeteksi", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()

