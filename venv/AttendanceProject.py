import cv2
import numpy
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "./ImagesAttendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def saveSignature(canvas):
    filename = f"Signature_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(filename, canvas)
    print(f"Signature saved as {filename}")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def markAttendance(name):
    filename = 'Attendance.csv'
    
    # Check if the file exists, and create it with headers if not
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('Name,Time\n')  # You can customize headers if needed

    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%Hh: %Mmin: %Ssec')
            f.writelines(f'\n{name}, {dtString}')


def findColor(img, myColors, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask, img)
        cv2.circle(img, (x, y), 10, myColorValues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x, y, count]) 
        count += 1
        # cv2.imshow(str(color[0]), mask)
    return newPoints

def getContours(img, imgOriginal):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(imgOriginal, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)

    return x+w//2, y

def drawOnCanvas(myPoints, myColorValues, canvas):
    for point in myPoints:
        cv2.circle(canvas, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)



myColors = [[5, 107, 0, 19, 255, 255], [160, 100, 100, 170, 255, 255], [57, 76, 0, 100, 255, 255]]
myColorValues = [[51, 153, 255],[255, 0, 255],[0, 255, 0]]
myPoints = [] #[x, y, colorId]

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

mode = 'face'  # Default mode is face recognition
signatureCanvas = None  # Placeholder for the canvas

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the webcam feed horizontally
    imgResult = img.copy()

    # Check the current mode
    if mode == 'signature':
        if signatureCanvas is None:
            signatureCanvas = np.zeros_like(img)  # Initialize the blank canvas
        # Drawing mode
        newPoints = findColor(img, myColors, myColorValues)
        if len(newPoints) != 0:
            for newP in newPoints:
                myPoints.append(newP)
        if len(myPoints) != 0:
            drawOnCanvas(myPoints, myColorValues, signatureCanvas)

        img = cv2.addWeighted(img, 0.5, signatureCanvas, 0.5, 0)

        cv2.putText(img, "Press 'c' to clear or 'q' to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif mode == 'face':
        # Face recognition mode
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = numpy.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, "Press 's' to start signature", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                markAttendance(name)

    # Display the webcam feed
    cv2.imshow('Webcam', img)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and mode == 'face':  # Switch to signature mode
        mode = 'signature'
        print("Switched to Signature Mode")
        signatureCanvas = np.zeros_like(img)  # Create a blank canvas for the signature
    elif key == ord('c') and mode == 'signature':  # Clear signature canvas
        myPoints = []
        signatureCanvas = np.zeros_like(img)  # Reset canvas
        print("Canvas cleared")
    elif key == ord('q') and mode == 'signature':  # Save the signature
        saveSignature(signatureCanvas)
        break  # Exit after saving
    elif key == ord('f'):  # Switch back to face mode
        mode = 'face'
        print("Switched to Face Recognition Mode")
    elif key == ord('x'):  # Quit the program
        break

cap.release()
cv2.destroyAllWindows()


# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255),2)
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255),2)
#
# results = face_recognition.compare_faces([encodeElon], encodeTest)
# faceDis = face_recognition.face_distance([encodeElon], encodeTest)