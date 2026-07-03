import cv2

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    print("Frame received:", ret)

    if not ret:
        print("Couldn't read frame.")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()