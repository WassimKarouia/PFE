import os
import pickle
import cv2
import mediapipe as mp


class HandLandmarkExtractor:
    def __init__(self, data_dir='./data', pickle_file='data.pickle'):
        self.data_dir = data_dir
        self.pickle_file = pickle_file
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    def extract_landmarks(self):
        data = []
        labels = []
        for dir_ in os.listdir(self.data_dir):
            for img_path in os.listdir(os.path.join(self.data_dir, dir_)):
                data_aux = []
                x_ = []
                y_ = []
                img = cv2.imread(os.path.join(self.data_dir, dir_, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
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

                    data.append(data_aux)
                    labels.append(dir_)

        with open(self.pickle_file, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)


if __name__ == "__main__":
    extractor = HandLandmarkExtractor()
    extractor.extract_landmarks()
