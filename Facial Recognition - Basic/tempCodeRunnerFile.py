#importing libraries and cascades
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray,frame):
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,105,180),3)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for(ex,ey,eh,ew) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh), (199,21,133), 2)
    return frame

Capture = cv2.VideoCapture(0)
while True:
    _, frame_img = Capture.read()
    gray_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray_img, frame_img)
    
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
Capture.release()
cv2.destroyAllWindows()
    

