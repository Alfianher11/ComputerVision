import cv2
import numpy as np

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame!")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    cx, cy = w // 2, h // 2

    # Ambil nilai warna di titik tengah
    b, g, r = frame[cy, cx]
    color_text = f"RGB: ({r}, {g}, {b})"

    # Gambar titik dan tampilkan nilai RGB
    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
    cv2.putText(frame, color_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (int(r), int(g), int(b)), 2)

    # Tampilkan kotak warna
    color_box = np.zeros((100, w, 3), dtype=np.uint8)
    color_box[:] = [b, g, r]

    # Gabungkan frame dan box
    output = np.vstack((frame, color_box))

    cv2.imshow("Deteksi Warna Kamera", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
