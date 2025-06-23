import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def smileDetect(gray,frame):
    #gray is the grayscale img frame is the color img
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,237,94), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        smile=smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_frame, (sx,sy), (sx+sw,sy+sh),(45,174,253), 2)
    return frame 

Capture =cv2.VideoCapture(0)

while True:
    _, frameimg = Capture.read()
    grayimg = cv2.cvtColor(frameimg, cv2.COLOR_BGR2GRAY)
    canvas = smileDetect(grayimg, frameimg)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Capture.release()
cv2.destroyAllWindows()
