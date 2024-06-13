from flask import Flask, render_template, request
import cv2
import os
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
from roboflow import Roboflow
from collections import Counter
from flask import jsonify
from flask_cors import CORS



from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

rf = Roboflow(api_key="lShrmkb3oqPEnjL0C8qP")
project = rf.workspace().project("model_vv")
model_rb = project.version(1).model

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('models/classif_model_V6.h5')

def decouper_et_enregistrer(image_path, output_folder, taille_fixe):
    image = cv2.imread(image_path)
    hauteur,largeur = image.shape[:2]
    taille_decoupe = taille_fixe
    sous_image_count = 1
    os.makedirs(output_folder, exist_ok=True)

    for x in range(0, largeur, taille_decoupe):
        sous_image = image[:, x:x+taille_decoupe]
        if sous_image.shape[1] < taille_decoupe:
            continue

        filename = os.path.basename(image_path)
        x=os.path.splitext(filename)[0]
        nom_sous_image = f"{x}_{sous_image_count}.png"
        chemin_sous_image = os.path.join(output_folder, nom_sous_image)
        cv2.imwrite(chemin_sous_image, sous_image)

        sous_image_count += 1

def preprocess_image(file_path,img_size):
    data = []
    try:
        img_arr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        data.append([resized_arr])
    except Exception as e:
        print(e)
    return np.array(data, dtype=object)

def delete_uploaded_files():
    existing_files_u = os.listdir('static/uploads')
    existing_files_c = os.listdir('static/cropped')
    existing_files_n = os.listdir('static/normal')
    for file in existing_files_u:
        os.remove(os.path.join('static/uploads', file))

    for file in existing_files_c:
        os.remove(os.path.join('static/cropped', file)) 

    for file in existing_files_n:
        os.remove(os.path.join('static/normal', file))       

def process_directory(directory_path,basepath):
    file_pred = []
    if os.listdir(directory_path):
        for image_name in os.listdir(directory_path):
            file_path_2 = os.path.join(basepath, directory_path, secure_filename(image_name)) 
            test = preprocess_image(file_path_2, 150)
            t_data = []
            for feature in test:
                t_data.append(feature)
            t_data = np.array(t_data) / 255  
            t_data = t_data.reshape(-1, 150, 150, 1)
            if t_data.dtype != np.float32:
                t_data = t_data.astype(np.float32)
            predicted_probabilities = model.predict(t_data)   
            prediction = np.argmax(predicted_probabilities, axis=1)[0]  
            if prediction == 0:
                label = "Defected"
            else:
                label = 'Normal'    
            file_pred.append((image_name, label))
    return file_pred

def defect_yl(file_path):
    prediction = model_rb.predict(file_path, confidence=1, overlap=30).json()
    if prediction['predictions']:
        #print(prediction)
        classes = [item['class'] for item in prediction['predictions']]
        class_counts = Counter(classes)
        occ_class = class_counts.most_common(1)
        #print(classes)
        #print(class_counts)
        name,class_cn=occ_class[0]
        #print(name)  
        return name 


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        delete_uploaded_files()
        images = request.files.getlist('imageInput')
        if not any(image for image in images):
            msg="no file uploaded"
            return jsonify({"msg": msg})
        i = 1
        for f in images:
            basepath = os.path.dirname(__file__)
            file_extension = os.path.splitext(f.filename)[1]
            new_filename = f.filename
            file_path = os.path.join(basepath, 'static/uploads', secure_filename(new_filename))
            f.save(file_path)
            #print(file_path)
            i += 1
            image = cv2.imread(file_path)
            hight,width = image.shape[:2]
            file_path_1 = os.path.join(basepath, 'static/normal', secure_filename(new_filename))
            if (width < 400):
                os.rename(file_path, file_path_1)
            else:
                decouper_et_enregistrer(file_path, 'static/cropped', 150)
            #print(width)
            #print(file_path)  

        file_pred_nr=[]  
        defected_nr_images=[]  
        if os.listdir('static/normal'):
            file_pred_nr=process_directory('static/normal',basepath)
            for image,label in file_pred_nr:
                if label=='Defected':
                    defected_nr_images.append(image)
        print(defected_nr_images)            
        #print(file_pred_nr)    
        
        file_pred_cr=[]
        defected_ss_images=[]
        defected_images=[]
        if os.listdir('static/cropped'):
            file_pred_cr=process_directory('static/cropped',basepath)
            for image,label in file_pred_cr:
                if label=='Defected':
                    defected_ss_images.append(image)
            for im in defected_ss_images:
                orig_img=im.split("_")[0]
                defected_images.append(orig_img+".png")        
        #print(file_pred_cr) 
        #remove redondance from the result
        #print(defected_ss_images)
        defected_images=list(set(defected_images)) 
            
        list_defection_nr=[]    
        for image in defected_nr_images:
            file_path_3 = os.path.join(basepath, 'static/normal', secure_filename(image))
            result=defect_yl(file_path_3)
            list_defection_nr.append((image,result))
        #print("normal",list_defection_nr)   
        translated_defection_nr = [(image, 'Déchirure') if result == 'dechire' else (image, 'Trou') if result == 'trou' else (image, result) for image, result in list_defection_nr]

        list_defection_cr=[]    
        for image in defected_ss_images:
            file_path_4 = os.path.join(basepath, 'static/cropped', secure_filename(image))
            result_1=defect_yl(file_path_4)
            list_defection_cr.append((image,result_1))
        modified_list = [(f"{image.split('_', 1)[0]}.png", result) for image, result in list_defection_cr]    
        #print(list_defection_cr)
        #print(modified_list)
        translated_defection_cr = [(f"{image.split('_', 1)[0]}", 'Déchirure') if result == 'dechire' else (f"{image.split('_', 1)[0]}", 'Trou') if result == 'trou' else (image, result) for image, result in modified_list]
        #print(translated_defection_cr)
        #print("cropped",modified_list)
        #print(defected_images)
        #print(list_defection_nr)     
                     

    return jsonify({"defected_images": defected_images, "defected_nr_images": defected_nr_images, "list_defection_nr":translated_defection_nr, "list_defection_cr":translated_defection_cr})

if __name__ == '__main__':
    app.run(debug=True)