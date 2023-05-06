import time
import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


# Path for exported data i.e numpy arrays
DATA_PATH = os.path.join('Numbers_Data')

images_per_sign = 200;

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# List of numbers from 0 to 9
# If you want this code to capture somthing else like A to z
# Just change the content of list and the DATA_PATH given above
num_list=[]
for i in range(0,10):
    num_list.append(str(i))

counter = 0

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

for number in num_list:
    counter=0
    folder_dir = os.path.realpath(os.path.join(DATA_PATH, str(number), ''))
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)

    while counter<200:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

                aspectRatio = h/w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal)/2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hGap + hCal, :] = imgResize

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

                cv2.putText(img, 'Collecting frames for number {}. Total captured images: {}'.format(number, counter), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            except Exception as e:
                continue
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder_dir}/Image_{time.time()}.jpg', imgWhite)
            print(counter)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()