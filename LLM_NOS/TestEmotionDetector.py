import cv2
import numpy as np
from keras.models import model_from_json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def load_emotion_model():
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/emotion_model.h5")
    print("Loaded emotion model from disk")
    return emotion_model

def detect_emotion(frame, emotion_model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emotion_data.append((timestamp, emotion_label))

        cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

def detect_humans(frame):
    human_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  

    return frame

def generate_pdf_report(emotion_data):
    pdf_filename = "emotion_report.pdf"
    pdf = canvas.Canvas(pdf_filename, pagesize=letter)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(300, 750, "Analysis Report")
    
    pdf.setFont("Helvetica", 12)

    pdf.setStrokeColorRGB(0, 0, 0)  
    pdf.setLineWidth(2)  
    pdf.rect(20, 20, 560, 750)

    for i, (timestamp, emotion_label) in enumerate(emotion_data, start=1):
        pdf.drawString(120, 710 - i * 20, f"{timestamp}: {emotion_label}")

    pdf.save()

def main():
    emotion_model = load_emotion_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break

        frame_with_humans = detect_humans(frame)

        detect_emotion(frame_with_humans, emotion_model)

        cv2.imshow('Detection', frame_with_humans)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            generate_pdf_report(emotion_data)
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion_data = []  
    main()