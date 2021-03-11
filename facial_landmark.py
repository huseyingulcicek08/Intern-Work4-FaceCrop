import cv2

# Read the input image
originalImage = cv2.imread('C:/Users/Grundig/Desktop/img-2.png')

# Convert into grayscale
gray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/Grundig/Desktop/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces and crop the faces
for (x, y, w, h) in faces:
    cv2.rectangle(originalImage, (x, y), (x + w, y + h),
                  (0, 0, 255), 2)

    faces = originalImage[y:y + h, x:x + w]
    cv2.imshow("CroppedFace", faces)
    cv2.imwrite('CroppedFace.jpg', faces)

# Display the output
cv2.imwrite('detcted.jpg', originalImage)
cv2.imshow('Original Image', originalImage)
cv2.waitKey()