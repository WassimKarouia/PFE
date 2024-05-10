import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def train_model(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model, x_test, y_test


def evaluate_model(model, x_test, y_test):
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly!'.format(score * 100))
    return score


class HandGestureClassifier:
    def __init__(self, data_file='./data.pickle', model_file='model.p'):
        self.data_file = data_file
        self.model_file = model_file

    def load_data(self):
        data_dict = pickle.load(open(self.data_file, 'rb'))
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])
        return data, labels

    def save_model(self, model):
        with open(self.model_file, 'wb') as f:
            pickle.dump({'model': model}, f)


if __name__ == "__main__":
    classifier = HandGestureClassifier()
    data, labels = classifier.load_data()
    trained_model, x_test, y_test = train_model(data, labels)
    accuracy = evaluate_model(trained_model, x_test, y_test)
    classifier.save_model(trained_model)
