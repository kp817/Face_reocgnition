# Face_reocgnition
using face_recognition module in python and open cv.
It is supervised learning model with labeled images as person's name on it.
Get some images from other sources and save them with person's name in a folder as img_detection.



First we will read images from img_detection folder and find the encodings of each image
        encodings: We need to extract a few basic measurments from each face.Then we could measure our unkonown face the same way and find the known face with the closest measurments.
The most reliable way to measure face is training the network to recognize picture objects like we did last time, we are going to train it to generate 128 measurments for each face
 training process involve:
        1. load a trining face image of a known person
        2. load another picture of the same known person
        3. load a picture of a totally different person
        
        
        
 Then we start reading pictures to recognize using webcam and start comparing pictures which are shown to cam with already in our dataset 
 It will return some values ranging from 0 to 1
      0: for not matched
      1: for matched sucessfully and recognize person and prints msg of that person.
