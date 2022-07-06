import os
from tkinter import *
import tkinter as tk
import webbrowser
import argparse
from tkinter import messagebox

from src.com_in_ineuron_ai_collect_trainingdata.get_faces_from_camera import TrainingDataCollector
from src.com_in_ineuron_ai_face_embedding.faces_embedding import GenerateFaceEmbedding
from src.com_in_ineuron_ai_training.train_softmax import TrainFaceRecogModel
from src.com_in_ineuron_ai_predictor.facePredictor import FacePredictor


class FaceRecognition:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Face Recognition")
        self.window.resizable(0, 0)
        self.window_height = 600
        self.window_width = 880

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        x_cordinate = int((screen_width / 2) - (self.window_width / 2))
        y_cordinate = int((screen_height / 2) - (self.window_height / 2))

        self.window.geometry("{}x{}+{}+{}".format(self.window_width, self.window_height, x_cordinate, y_cordinate))
        self.window.configure(background='#ffffff')

        text = tk.Label(self.window, text="Face Recognition", font=("Arial Bold", 30), bg='#ffffff')
        text.place(x=250, y=50)
        self.Register_entry = tk.Entry(self.window, font=("Segoe UI", 15))
        self.Register_entry.place(x=200, y=150)
        Register = tk.Button(self.window, text="Register", fg="white", bg="#363e75", command=self.Register,
                             font=('times', 18, 'bold', 'underline'), width=12)
        Register.place(x=200, y=200)
        self.Recognize_entry = tk.Entry(self.window, font=("Segoe UI", 15))
        self.Recognize_entry.place(x=420, y=150)
        Recognize = tk.Button(self.window, text="Recognize", fg="white", bg="#363e75",command=self.authenticate,
                              font=('times', 18, 'bold', 'underline'), width=12)
        Recognize.place(x=420, y=200)
        text = tk.Label(self.window, text="Web Authentication", font=("Arial Bold", 30), bg='#ffffff')
        text.place(x=250, y=300)
        button1 = tk.Button(self.window, text="Register Web", width=12, fg="white", bg="#363e75",
                            font=('times', 18, 'bold', 'underline'),
                            command=lambda: webbrowser.open_new("http://localhost:5000/register"))
        button1.place(x=200, y=400)
        button2 = tk.Button(self.window, text="Recognize Web", width=12, fg="white", bg="#363e75",
                            font=('times', 18, 'bold', 'underline'),
                            command=lambda: webbrowser.open_new("http://localhost:5000/check"))
        button2.place(x=420, y=400)
        button3 = tk.Button(self.window, text="Exit", fg="white", bg="#363e75",
                            font=('times', 18, 'bold', 'underline'), command=self.window.destroy)
        button3.place(x=370, y=500)

        self.window.mainloop()

    def collectUserImageForRegistration(self, imageSaveLocation):
        ap = argparse.ArgumentParser()

        ap.add_argument("--faces", default=50,
                        help="Number of faces that camera will get")
        ap.add_argument("--output", default="../datasets/train/" + imageSaveLocation,
                        help="Path to faces output")

        args = vars(ap.parse_args())

        trnngDataCollctrObj = TrainingDataCollector(args)
        trnngDataCollctrObj.collectImagesFromCamera()

    def getFaceEmbedding(self):

        ap = argparse.ArgumentParser()

        ap.add_argument("--dataset", default="../datasets/train",
                        help="Path to training dataset")
        ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle")
        # Argument of insightface
        ap.add_argument('--image-size', default='112,112', help='')
        ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
        ap.add_argument('--ga-model', default='', help='path to load model.')
        ap.add_argument('--gpu', default=0, type=int, help='gpu id')
        ap.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
        args = ap.parse_args()

        genFaceEmbdng = GenerateFaceEmbedding(args)
        genFaceEmbdng.genFaceEmbedding()

    def trainModel(self):
        # ============================================= Training Params ====================================================== #

        ap = argparse.ArgumentParser()
        ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle",
                        help="path to serialized db of facial embeddings")
        ap.add_argument("--model", default="faceEmbeddingModels/my_model.h5",
                        help="path to output trained model")
        ap.add_argument("--le", default="faceEmbeddingModels/le.pickle",
                        help="path to output label encoder")

        args = vars(ap.parse_args())

        faceRecogModel = TrainFaceRecogModel(args)
        faceRecogModel.trainKerasModelForFaceRecognition()

    def get_reg_names(self):
        # get folder names from the folder containing the registered faces
        registered_names = []
        registered_names_path = "../datasets/train"
        for name in os.listdir(registered_names_path):
            registered_names.append(name)
        return registered_names

    def Register(self):
        self.collectUserImageForRegistration(self.Register_entry.get())
        self.getFaceEmbedding()
        self.trainModel()

    def authenticate(self):
        faceDetector = FacePredictor()
        name_check = faceDetector.detectFace()
        if name_check == self.Recognize_entry.get():
            # toast success
            messagebox.showinfo("Success", "Welcome " + name_check)
        else:
            messagebox.showinfo("Error", "Please try again")


if __name__ == '__main__':
    call = FaceRecognition()