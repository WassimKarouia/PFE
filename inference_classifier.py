import cv2
import mediapipe as mp
import numpy as np
import customtkinter
from customtkinter import *
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import pickle
import pyttsx3
import time


class Camera:
    def __init__(self, label, letters_label, interface):
        self.camera_open = True
        self.label = label
        self.letters_label = letters_label
        self.interface = interface
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.labels_dict = {0: 'Merci', 1: 'Derien', 2: 'aide', 3: 'D', 4: 'E',
                            5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
                            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
                            25: 'Z', 26: ' '}
        self.detected_letters = []

    def open_camera_thread(self):
        while self.camera_open:
            data_aux = []
            x_ = []
            y_ = []
            self.detected_letters = []
            ret, frame = self.cap.read()
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

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

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                if len(data_aux) == 84:
                    data_aux = data_aux[:42]

                prediction = self.interface.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]
                self.detected_letters.append(predicted_character)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                self.interface.build_phrase(self.detected_letters)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)

            self.label.config(image=frame)
            self.label.image = frame

            self.letters_label.config(text=" ".join(self.detected_letters))
            self.letters_label.place(relx=0.2, rely=0.70)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


class InterfaceUtilisateur:
    def __init__(self, fenetre, model):
        self.engine = pyttsx3.init()
        self.fenetre = fenetre
        self.fenetre.geometry('1280x720')
        self.fenetre.title('Convertir la langage des signes en texte et voix')
        self.fenetre['bg'] = '#E2E9C0'
        self.fenetre.resizable(height=False, width=False)




        # Création des cadres pour diviser l'interface
        self.frame_gauche = Frame(self.fenetre, bg='#313131')
        self.frame_gauche.place(relx=0, rely=0, relwidth=0.7, relheight=1)

        self.frame_droite = Frame(self.fenetre, bg='#282828')
        self.frame_droite.place(relx=0.7, rely=0, relwidth=0.3, relheight=1)

        # Ajout des éléments à la partie gauche (caméra, lettres, phrase)
        self.camera_label = Label(self.frame_gauche, bg='#e5ddde', relief=SOLID, highlightbackground='red',
                                  highlightcolor='green')
        self.camera_label.place(x=0, y=0, width=620, height=420)
        self.hands_label = Label(self.camera_label, bg='white')
        self.hands_label.pack(side=RIGHT, padx=5, pady=5)

        self.letters_label = Label(self.frame_gauche, text="", bg='#313131',
                                   font=("Helvitica", 12), fg="#e5ddde")
        self.letters_label.place(relx=0.05, rely=0.70)

        self.label_letters = Label(self.frame_gauche, text="Letters : ", bg="#313131",
                                   font=("Helvitica", 12, "bold"), fg="#e5ddde")
        self.label_letters.place(relx=0.02, rely=0.70)

        self.label_phrase = Label(self.frame_gauche, text="Phrase : ", bg="#313131",
                                  font=("Helvitica", 12, "bold"), wraplength=300, fg="#e5ddde")
        self.label_phrase.place(relx=0.02, rely=0.80)

       

        # Ajout des éléments à la partie droite (boutons)

        self.bouton_clear = customtkinter.CTkButton(master=self.frame_droite, text="Clear",
                                                    command=self.clear_interface)
        self.bouton_clear.place(relx=0.5, rely=0.2, anchor=CENTER)

        self.bouton_speak = customtkinter.CTkButton(self.frame_droite, text="Audio",
                                                    command=self.speak_phrase)
        self.bouton_speak.place(relx=0.5, rely=0.4, anchor=CENTER)


        self.bouton_delete = customtkinter.CTkButton(self.frame_droite, text="DEL",
                                                     command=self.delete_letter)
        self.bouton_delete.place(relx=0.5, rely=0.6, anchor=CENTER)



        self.model = model

        self.camera = Camera(self.camera_label, self.letters_label, self)
        self.camera_thread = threading.Thread(
            target=self.camera.open_camera_thread)
        self.camera_thread.start()

        self.phrase_text = ""
        self.last_time_letter_added = time.time()
        self.previous_phrase_length = 0
        self.correction_dict = {
            'H': ['B', 'G'],
            'J': ['I'],
        }

    def build_phrase(self, detected_letters):
        current_time = time.time()
        if current_time - self.last_time_letter_added >= 2:
            for i, letter in enumerate(detected_letters):
                corrected_letter = self.autocorrect_letter(letter)
                self.phrase_text += corrected_letter
            self.last_time_letter_added = current_time

        phrase_lines = [self.phrase_text[i:i + 30] for i in range(0, len(self.phrase_text), 30)]
        phrase_text_wrapped = "\n".join(phrase_lines)

        self.label_phrase.config(text="Phrase : " + phrase_text_wrapped,
                                 wraplength=1000)
        self.label_phrase.place(relx=0.02, rely=0.80)

    def clear_interface(self):
        previous_length = len(self.phrase_text)

        self.phrase_text = self.phrase_text[:self.previous_phrase_length]

        self.previous_phrase_length = len(self.phrase_text)

        self.label_phrase.config(text="Phrase : " + self.phrase_text)

        self.fenetre.focus_set()

    def speak_phrase(self):
        # Réglez la vitesse de la parole
        self.engine.setProperty('rate', 150)  # 150 mots par minute

        # Utilisez une voix plus claire et naturelle si disponible
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "french" in voice.languages:
                self.engine.setProperty('voice', voice.id)
                break

        # Ajoutez une pause après chaque phrase pour une meilleure compréhension
        self.engine.say(self.phrase_text)
        self.engine.runAndWait()

    def update_phrase_periodically(self):
        detected_letters = self.camera.detected_letters
        if detected_letters:
            self.build_phrase(detected_letters)

        self.fenetre.after(2000, self.update_phrase_periodically)

    def delete_letter(self):
        if self.phrase_text:
            self.phrase_text = self.phrase_text[:-1]

            self.label_phrase.config(text="Phrase : " + self.phrase_text)

    def autocorrect_letter(self, letter):
        if letter in self.correction_dict:
            possible_corrections = self.correction_dict[letter]
            most_common_correction = max(set(possible_corrections), key=possible_corrections.count)
            return most_common_correction
        else:
            return letter


class ModelePrediction:
    def __init__(self, model_path):
        model_dict = pickle.load(open(model_path, 'rb'))
        self.model = model_dict['model']

    def predict(self, data):
        return self.model.predict(data)


if __name__ == "__main__":
    modele_prediction = ModelePrediction('./model.p')
    fenetre = Tk()
    interface_utilisateur = InterfaceUtilisateur(fenetre, modele_prediction)
    fenetre.mainloop()

