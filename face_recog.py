import face_recognition
import cv2
import numpy as np
import os 

path='img_detection'
images=[]
class_names=[]
mylist=os.listdir(path)
print(mylist)
for cls in mylist:
    curimg=cv2.imread(f'{path}/{cls}')
    images.append(curimg)
    class_names.append(os.path.splitext(cls)[0])
print(class_names)

def find_encode(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown=find_encode(images)
print('encoding completed')

cap=cv2.VideoCapture(0)

while True:
    sucess,img=cap.read()
    imgsmall=cv2.resize(img,(0,0),None,0.25,0.25)
    imgsmall=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    facescurrframe=face_recognition.face_locations(imgsmall)
    encodecurrframe=face_recognition.face_encodings(imgsmall,facescurrframe)
    
    
    for encodeface,faceloc in zip(encodecurrframe,facescurrframe):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        facedis=face_recognition.face_distance(encodelistknown,encodeface)
        print(facedis)
        matchindex=np.argmin(facedis)
        
        if matches[matchindex]:
            name=class_names[matchindex].upper()
            print(name)
            y1,x2,y2,x1=faceloc
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(img,name,(x1+6,y2+12),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            
    cv2.imshow('webcam',img)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cv2.destroyAllWindows()