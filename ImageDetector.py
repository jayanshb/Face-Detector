import cv2,sys

# the file path would be passes as an argument into the script
source ="image.jpg"
image = cv2.imread(source)

#converting to grayscale for better image recognition
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#creating a faceCascade classifier in order for face recogntion 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors = 3,minSize=(30,30))

print("Number of faces found is "+str(len(faces)))

# create rectangles around the detected faces 
for x,y,w,h in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    roi_color = image[y:y+h,x:x+w]
    print("Image found. Saving locally")
    cv2.imwrite(str(w)+str(h)+ '_faces.jpg',roi_color)

# writing the detected faces image into local directory
status = cv2.imwrite('faces_detected.jpg',image)

print("ststus of writing files:"+str(status))

# --- writing seperate images in the detected faces image into local directory





