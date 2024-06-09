# import tensorflow as tf
import os
import cv2
from mtcnn import MTCNN
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('Large_F3.h5')
class_names = [
    '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2',
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41',
    '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '6', '7', 
    '8', '9'
]
def predictor(image):
    # print('yess')
    ans = model.predict(image)
    # print('uess')
    final_ans = class_names[np.argmax(ans)]
    print(final_ans)
    data = pd.read_excel("Data clean 2303.xlsx")
    # print('yess') 
    p_data = data.iloc[[int(final_ans)-1]]
    # print(p_data)
    return p_data


def check(image, padding=60):
    detector = MTCNN()
    
    img = cv2.imread(image)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = detector.detect_faces(img_rgb)
    
    if not faces:
        print("No faces detected in the image.")
        return
    
    face = faces[0]
    x, y, w, h = face['box']
    
    x_pad = max(0, x - padding)
    y_pad = max(0, y - padding)
    w_pad = w + 2 * padding
    h_pad = h + 2 * padding
    
    # Ensure the padded box is within image boundaries
    x_end = min(img_rgb.shape[1], x_pad + w_pad)
    y_end = min(img_rgb.shape[0], y_pad + h_pad)
    
    face_img = img_rgb[y_pad:y_end, x_pad:x_end]
    
    resized_face_img = cv2.resize(face_img, (256, 256))
    
    result = predictor(np.expand_dims(resized_face_img, axis=0))
    
    plt.imshow(resized_face_img)
    plt.title("Resized Face Image with Padding")
    plt.axis('off')
    plt.show()

    return result



""" 
def check(image, padding=60):
    detector = MTCNN()
    img = cv2.imread(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image using MTCNN
    faces = detector.detect_faces(img_rgb)
    if not faces:
        print("No faces detected in the image.")
        return
    face = faces[0]
    # print(faces[0])
    x, y, w, h = face['box']
    face_img = img_rgb[y:y+h, x:x+w]
    resized_face_img = cv2.resize(face_img, (256, 256))
    result = predictor(np.expand_dims(resized_face_img, axis=0))
    # print(result)
    return result
"""