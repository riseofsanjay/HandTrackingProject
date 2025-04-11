import cv2
from cvzone.HandTrackingModule import HandDetector

# Webcam setup (try 0, 1, or use CAP_DSHOW for Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

# Check if webcam opens successfully
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam. Try changing the camera index or check permission.")

# Load image and make it smaller
img1 = cv2.imread("img_1.png")
if img1 is None:
    raise ValueError("❌ Image 'img_1.png' not found in your folder!")

img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2))  # Half size

# Hand detector
detector = HandDetector(detectionCon=0.8)

# Variables for zoom
startDist = None
scale = 0
cx, cy = 640, 360  # initial center

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from webcam")
        break

    hands, img = detector.findHands(img)  # With drawing

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        if len(lmList) >= 9:
            # Thumb tip and index tip positions
            x1, y1 = lmList[4][:2]
            x2, y2 = lmList[8][:2]

            # Distance between thumb and index
            length, info, img = detector.findDistance((x1, y1), (x2, y2), img)

            if startDist is None:
                startDist = length

            scale = int((length - startDist) // 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center between fingers

    else:
        startDist = None

    # Resize the image according to scale
    h1, w1, _ = img1.shape
    newH, newW = h1 + scale, w1 + scale

    # Avoid negative or zero dimensions
    newH = max(1, (newH // 2) * 2)
    newW = max(1, (newW // 2) * 2)

    img1_resized = cv2.resize(img1, (newW, newH))

    # Place the image at the center point
    h, w, _ = img.shape
    x1, x2 = max(0, cx - newW // 2), min(w, cx + newW // 2)
    y1, y2 = max(0, cy - newH // 2), min(h, cy + newH // 2)

    try:
        img[y1:y2, x1:x2] = img1_resized[:y2 - y1, :x2 - x1]
    except:
        pass  # Avoid crash if dimensions go out of bounds

    # Show zoom level
    zoom_percent = int((newW / w1) * 100)
    cv2.putText(img, f"Zoom: {zoom_percent}%", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    cv2.imshow("Hand Zoom Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
