import cv2
import mediapipe as mp

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Bagian tubuh utama (landmark ID dan nama label)
bagian_tubuh = {
    0: 'Kepala (Hidung)',
    11: 'Bahu Kiri',
    12: 'Bahu Kanan',
    13: 'Siku Kiri',
    14: 'Siku Kanan',
    15: 'Pergelangan Kiri',
    16: 'Pergelangan Kanan',
    23: 'Pinggul Kiri',
    24: 'Pinggul Kanan',
    25: 'Lutut Kiri',
    26: 'Lutut Kanan',
    27: 'Mata Kaki Kiri',
    28: 'Mata Kaki Kanan'
}

# Kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        for id, nama in bagian_tubuh.items():
            landmark = result.pose_landmarks.landmark[id]
            x, y = int(landmark.x * w), int(landmark.y * h)

            # Gambar kotak kecil di sekitar titik
            cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), (0, 255, 0), 2)
            # Tampilkan label
            cv2.putText(frame, nama, (x+12, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Tandai Bagian Tubuh", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC
        break

cap.release()
cv2.destroyAllWindows()
