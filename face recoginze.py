import cv2
import os
import numpy as np
from tkinter import Tk, Label, Entry, Button, StringVar
from tkinter.ttk import Style
from tkinter import messagebox

# Ensure the 'faces' directory exists
if not os.path.exists('faces'):
    os.makedirs('faces')

# Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Variables to store training data
IMAGE_SIZE = (200, 200)  # Standard size for all face images
labels = []
images = []

# Function to capture a face
def capture_face():
    name = name_entry.get().strip()
    if not name:
        status_var.set("Please enter a name.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Press "s" to save your face image', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cropped_face = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(cropped_face, IMAGE_SIZE)
                face_path = f'faces/{name}_{len(os.listdir("faces"))}.jpg'
                cv2.imwrite(face_path, resized_face)
                status_var.set(f"Image saved as {face_path}")
            else:
                status_var.set("No face detected. Try again.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to load training data
def load_training_data():
    global images, labels
    images = []
    labels = []
    label_map = {}

    for file_name in os.listdir('faces'):
        if file_name.endswith('.jpg'):
            name = file_name.split('_')[0]
            if name not in label_map:
                label_map[name] = len(label_map)

            label = label_map[name]
            image_path = os.path.join('faces', file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, IMAGE_SIZE)

            images.append(resized_image)
            labels.append(label)

    if images:
        face_recognizer.train(images, np.array(labels))

    return label_map

# Function to recognize faces
def recognize_faces():
    if not os.listdir('faces'):
        status_var.set("No saved faces to recognize.")
        return

    label_map = load_training_data()
    reverse_label_map = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            resized_face_roi = cv2.resize(face_roi, IMAGE_SIZE)

            label, confidence = face_recognizer.predict(resized_face_roi)

            name = reverse_label_map.get(label, "Unknown")
            text = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Recognizing Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Setup
root = Tk()
root.title("Face Capture and Recognition")
root.geometry("400x200")

style = Style()
style.configure("TButton", font=("Helvetica", 12))
style.configure("TLabel", font=("Helvetica", 12))

# GUI Components
Label(root, text="Enter your name:").pack(pady=10)
name_entry = Entry(root, font=("Helvetica", 12))
name_entry.pack(pady=5)

status_var = StringVar()
status_label = Label(root, textvariable=status_var, foreground="green")
status_label.pack(pady=10)

Button(root, text="Capture Image", command=capture_face).pack(pady=5)
Button(root, text="Recognize Faces", command=recognize_faces).pack(pady=5)

# Start the GUI loop
root.mainloop()
