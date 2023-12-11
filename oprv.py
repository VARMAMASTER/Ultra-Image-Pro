import cv2
import numpy as np
from keras.models import load_model

model = load_model('./Ultrasonic_welding_IPDL.h5')
classes = ['dent', 'good', 'overextrusion', 'scratch']

def process_frame(frame):
   
    img = cv2.resize(frame, (224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    max_predict = np.argmax(prediction)
    defect = classes[max_predict]
    
    return defect

cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    edges = cv2.Canny(blurred_frame, 100, 200)
    combined_frame = cv2.addWeighted(frame, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)
    mirrored_frame = cv2.flip(combined_frame, 1)
    defect = process_frame(mirrored_frame)
    cv2.putText(mirrored_frame, f"Defect: {defect}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Enhanced Camera Feed', mirrored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
