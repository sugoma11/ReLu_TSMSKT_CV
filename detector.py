import cv2
from utils.eyes_cropper import eye_cropper
eye_model = keras.models.load_model('best_model.h5')


cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(cap.get(cv2.CAP_PROP_FPS))
if not cap.isOpened():
    raise IOError("Cannot open webcam")

counter = 0
# create a while loop that runs while webcam is in use
while True:
    # capture frames being outputted by webcam
    ret, frame = cap.read()
    # function called on the frame
    image_for_prediction = eye_cropper(frame)
    try:
       image_for_prediction = image_for_prediction/255.0
    except BaseException:
       continue

    prediction = eye_model.predict(image_for_prediction)
    if prediction < 0.5:
        counter = 0
        status = "Open"
        cv2.putText(frame, status, (round(w / 2) - 80, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)

    else:
        counter = counter + 1
        status = "Closed"
        cv2.putText(frame, status, (round(w / 2) - 104, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_4)

    if counter > 5:
        cv2.putText(frame, "DRIVER SLEEPING", (round(w / 2) - 136, round(h) - 146), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        counter = 5

    cv2.imshow("Drowsiness Detection", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
