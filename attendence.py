from csv import writer
import csv
from unittest.mock import patch
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Images'
images = []
person_name = []
my_list = os.listdir(path)
print(my_list)
print("Type of mylist = ",type(my_list))
x=my_list.pop(my_list.index("Group.jpg"))
print("New List:",my_list,"\nGroup Image:",x)
grp_img=cv2.imread(f'{path}/{x}')

for cu_img in my_list:
    current_img=cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    person_name.append(os.path.splitext(cu_img)[0])
print(person_name)

def faceEncodings(images):
    # Encoding means gets 128 facial features
    # uses HOG (HISTOGRAM OF ORIENTED GRADIENTS) algorithm by default alternate option cnn
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

def attendance(name):
    with open('Attendence.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        csvwriter = csv.writer(f)
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        
        if name not in name_list:
            time_now = datetime.now()
            tstr = time_now.strftime('%H:%M:%S')
            dstr = time_now.strftime('%d-%m-%Y')
            data=[name,tstr,dstr]
            csvwriter.writerow(data)

encode_list_known = faceEncodings(images)

# face_loc = face_recognition.face_locations(grp_img,1,"cnn")
face_loc = face_recognition.face_locations(grp_img)
grp_encode = face_recognition.face_encodings(grp_img,face_loc)
print("All encodings Completed!!!!")

for person,face in zip(grp_encode,face_loc):
    matches = face_recognition.compare_faces(encode_list_known,person)
    face_dist = face_recognition.face_distance(encode_list_known,person)
    match_index = np.argmin(face_dist)
    if matches[match_index]:
        name = person_name[match_index].upper()
        # print(name)
        y1,x2,y2,x1 = face
        # print(face)
        cv2.rectangle(grp_img, (x1,y1),(x2,y2), (0,255,0),1)
        cv2.rectangle(grp_img, (x1, y2-10),(x2,y2),(0,255,0), cv2.FILLED)
        cv2.putText(grp_img, name, (x1+4, y2-3), cv2.FONT_ITALIC, 1/3, (0,0,255),1)
        attendance(name)

cv2.imshow("Result group image",grp_img)
cv2.waitKey(0)

# print("GRP encodings:\n",grp_encode[0])

# =========================================================================================================

# cap = cv2.VideoCapture(0)   # parameter for inbuilt cam = 0 & parameter for external web cam = 1
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1100)
# while True:
#     ret, frame = cap.read()
#     faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
#     faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

#     faces_current_frame = face_recognition.face_locations(faces)
#     encodes_current_frame = face_recognition.face_encodings(faces, faces_current_frame)
    
#     for encodeFace, faceLoc in zip(encodes_current_frame, faces_current_frame):
#         matches = face_recognition.compare_faces(encode_list_known, encodeFace)
#         face_dis = face_recognition.face_distance(encode_list_known, encodeFace)

#         match_index = np.argmin(face_dis)

#         if matches[match_index]:
#             name = person_name[match_index].upper()
#             # print(name)
#             y1,x2,y2,x1 = faceLoc
#             y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0),2)
#             cv2.rectangle(frame, (x1, y2-35),(x2,y2),(0,255,0), cv2.FILLED)
#             cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
#             attendance(name)

#     cv2.imshow("Camera", frame)
#     if cv2.waitKey(10) == 13:
#         break
# cap.release()
# cv2.destroyAllWindows()