import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_count=0
    hand_frame_coordinates=[]

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        hand_count =0
        for hand_landmarks in results.multi_hand_landmarks:
            x_=[]
            y_=[]
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            hand_count+=1

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            hand_frame_coordinates.append(list([x1,y1,x2,y2]))

        predictedChars=[]
        totalCodsForOneHand = 42
        startCords=0
        endCords=42
        for i in range(hand_count):
            handCords = data_aux[startCords:endCords]
            startCords+=totalCodsForOneHand
            endCords+=totalCodsForOneHand
            hand_prediction = model.predict([np.asarray(handCords)])
            predicted_character = labels_dict[int(hand_prediction[0])]
            predictedChars.append(predicted_character)



        for i in range(hand_count):
            coordinates = hand_frame_coordinates[i]
            x1= coordinates[0]
            y1= coordinates[1]
            x2= coordinates[2]
            y2= coordinates[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            cv2.putText(frame, predictedChars[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
