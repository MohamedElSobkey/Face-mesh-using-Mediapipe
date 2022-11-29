import cv2  
import mediapipe as mp

img = cv2.imread('person.jpg')



#face_ mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

rgb_img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

# Facial Landmarks

res_img = mp_face_mesh.process(rgb_img)

h , w , _ = img.shape

for facial_landmarks in res_img.multi_face_landmarks:
     for i in range (0,468) :
        pt1 = facial_landmarks.landmark[i]
        print (pt1)
        x = int (pt1.x * w)
        y = int (pt1.y * h)
        cv2.circle(img, (x,y), 2 , (100,100, 0), -1)
        
        
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


####################

#reeeeeeeeeaaaaaaaaaaallllllllll TIME

import cv2  
import mediapipe as mp

#face_ mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while True :
    
    ret, img = cap.read()
    if ret is not True :
         break

    h , w , _ = img.shape
    
    rgb_img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    
     # Facial Landmarks
    
    res_img = mp_face_mesh.process(rgb_img)
    
   
    for facial_landmarks in res_img.multi_face_landmarks:
         for i in range (0,468) :
            pt1 = facial_landmarks.landmark[i]
            print (pt1)
            x = int (pt1.x * w)
            y = int (pt1.y * h)
            cv2.circle(img, (x,y), 2 , (100,100, 0), -1)
            
            
    cv2.imshow('img', img)
    if cv2.waitKey(1) ==ord('q'):
        cv2.destroyAllWindows()
        break
    
cap.release()
cv2.destroyAllWindows() 



        
        
        

