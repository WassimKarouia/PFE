import os
import cv2


class DataCollector:
    def __init__(self, data_dir='./data', number_of_classes=26, dataset_size=100):
        self.data_dir = data_dir
        self.number_of_classes = number_of_classes
        self.dataset_size = dataset_size

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.cap = cv2.VideoCapture(0)

    def collect_data(self):
        for j in range(self.number_of_classes):
            class_dir = os.path.join(self.data_dir, str(j))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            print('Collecting data for class {}'.format(j))

            self._wait_for_key_press()

            self._collect_class_data(j)

    def _wait_for_key_press(self):
        done = False
        while not done:
            ret, frame = self.cap.read()
            cv2.putText(frame, 'Ready? Press "R" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                done = True

    def _collect_class_data(self, class_index):
        counter = 0
        while counter < self.dataset_size:
            ret, frame = self.cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            cv2.imwrite(os.path.join(self.data_dir, str(class_index), '{}.jpg'.format(counter)), frame)
            counter += 1

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()
    collector.release()
