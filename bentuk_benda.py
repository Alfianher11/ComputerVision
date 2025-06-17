import cv2
import numpy as np

def detect_shape(contour):
    # Hitung keliling kontur
    peri = cv2.arcLength(contour, True)
    # Approximate kontur (kurangi detail)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    sides = len(approx)

    # Identifikasi bentuk
    if sides == 3:
        return "Segitiga"
    elif sides == 4:
        # Cek apakah persegi atau persegi panjang
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Persegi"
        else:
            return "Persegi Panjang"
    elif sides == 5:
        return "Segilima"
    elif sides == 6:
        return "Segienam"
    else:
        area = cv2.contourArea(contour)
        if area > 100 and sides > 6:
            return "Lingkaran"
        return "Tak dikenal"

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter kontur kecil
            shape = detect_shape(cnt)
            # Gambar kontur
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            # Tulis nama bentuk
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.putText(frame, shape, (cx - 50, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Deteksi Bentuk Benda", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()