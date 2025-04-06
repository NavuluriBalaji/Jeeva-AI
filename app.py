from os import path
# from flask import Flask
from flask import Flask, render_template, request, jsonify, redirect, flash, url_for, send_file, make_response, send_from_directory
import firebase_admin
from transformers import pipeline
from firebase_admin import credentials, auth, exceptions
import pickle
import joblib
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
import numpy as np
from firebase_admin import credentials, auth, exceptions
import os
from PIL import Image
from fpdf import FPDF
import requests
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

app.secret_key = 'ebcef35de0a1ea35afa1541de8e92573'  # Ensure this line is present and set to a unique and secret value
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Mapps API Key
MAPPS_API_KEY = '074994fb1cc7a26816f77cca2ab6534d'

# Define pipelines for each disease
pipelines = {
    "brain_tumor": pipeline("image-classification", model="Devarshi/Brain_Tumor_Classification"),
    "breast_cancer": pipeline("image-classification", model="amanvvip2/finetuned-breast_cancer_images"),
    "skin_cancer": pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification"),
    "lung_colon_cancer": pipeline("image-classification", model="DunnBC22/vit-base-patch16-224-in21k_lung_and_colon_cancer")
    # Remove the unrecognized model
    # "whole_body_CT": pipeline("image-segmentation", model="project-lighter/whole_body_segmentation")
}

# Define OCR pipeline
# ocr_pipeline = pipeline("image-to-text", model="jinhybr/OCR-Donut-CORD")
ocr_pipeline = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# # Initialize Firebase
# cred = credentials.Certificate("admin_sdk.json")
# firebase_admin.initialize_app(cred)

sym_des = pd.read_csv("flask/datasets/symtoms_df.csv")
precautions = pd.read_csv("flask/datasets/precautions_df.csv")
workout = pd.read_csv("flask/datasets/workout_df.csv")
description = pd.read_csv("flask/datasets/description.csv")
medications = pd.read_csv('flask/datasets/medications.csv')
diets = pd.read_csv("flask/datasets/diets.csv")

model = pickle.load(open('flask/svc.pkl', 'rb'))

# Load models for kidney, heart, and diabetes predictions
# diabetes_model = pickle.load(open('flask/diabetes.pkl', 'rb'))
heart_disease_model = pickle.load(open('flask/heart.pkl', 'rb'))
kidney_disease_model = pickle.load(open('flask/kidney.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

# def verify_password(uid, password):
#     # Implement your password verification logic here
#     # For example, you can store hashed passwords in Firebase and compare them
#     return True

@app.route('/assistant')
def assistant():
    return render_template('assistant.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

@app.route('/recommend')
def recommend():
    return render_template('old_recommend.html')

@app.route('/reminder')
def reminder():
    return render_template('reminder.html')

@app.route('/related')
def related():
    return render_template('related.html')

@app.route('/appointment')
def appointment():
    doctors = ["Dermatologist", "Allergist", "Gastroenterologist", "Hepatologist", "Osteopathic", "Endocrinologist", "Pulmonologist", "Cardiologist", "Neurologist", "Internal Medicine", "Pediatrician", "Common Cold", "Cardiologist", "Phlebologist", "Osteoarthristis", "Rheumatologists", "Otolaryngologist", "Dermatologists", "Gynecologist"]
    return render_template('appointment.html', doctors=doctors)

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    symptoms = list(symptoms_dict.keys())
    return jsonify(symptoms)

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {
    0: '(vertigo) Paroymsal Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer disease',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'hepatitis A'
}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
        else:
            print(f"Warning: Symptom '{item}' not found in symptoms_dict")
    prediction = model.predict([input_vector])[0]
    print(f"Prediction: {prediction}, Type: {type(prediction)}")  # Debugging statement
    if isinstance(prediction, int):
        if prediction in diseases_list:
            return diseases_list[prediction]
        else:
            raise ValueError(f"Predicted index '{prediction}' not found in diseases_list")
    elif isinstance(prediction, str):
        return prediction
    else:
        raise ValueError(f"Unexpected prediction type: {type(prediction)}")

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('prediction.html', message=message)
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('prediction.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('prediction.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files or 'model' not in request.form:
        return 'No file or model selected', 400

    file = request.files['file']
    model = request.form['model']

    if file.filename == '':
        return 'No selected file', 400

    if file and model in pipelines:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return 'Invalid file format. Only .png, .jpg, .jpeg are allowed.', 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load the image and classify
        image = Image.open(filepath)
        prediction = pipelines[model](image)

        # Cleanup uploaded file
        os.remove(filepath)

        # Return result
        return render_template('visresult.html', prediction=prediction, model=model)

    return 'Invalid request', 400

@app.route('/visual')
def visual():
    return render_template('visual.html')

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    name = request.form.get('name')
    age = request.form.get('age')
    phone = request.form.get('phone')
    gender = request.form.get('gender')
    date = request.form.get('date')
    disease = request.form.get('disease')
    doctor = request.form.get('doctor')

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="HealthCare Appointment Confirmation", ln=True, align='C')
    pdf.ln(10)  # Add a line break

    pdf.set_font("Arial", size=12)
    pdf.cell(100, 10, txt=f"Name: {name}", ln=False)
    pdf.cell(100, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(100, 10, txt=f"Phone: {phone}", ln=False)
    pdf.cell(100, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(100, 10, txt=f"Appointment Date: {date}", ln=False)
    pdf.cell(100, 10, txt=f"Disease: {disease}", ln=True)
    pdf.cell(100, 10, txt=f"Doctor: {doctor}", ln=True)
    pdf.ln(10)  # Add a line break

    pdf.cell(200, 10, txt="Money Received: [ ]", ln=True)
    pdf.ln(10)  # Add a line break

    pdf.cell(200, 10, txt="Signature: ___________________________", ln=True)
    pdf.ln(20)  # Add a line break

    pdf.cell(200, 10, txt="Doctor's Prescription:", ln=True)
    pdf.ln(10)  # Add a line break
    pdf.multi_cell(0, 10, txt=" " * 200)  # Add space for the doctor's prescription

    pdf_output = f"./uploads/{name}_appointment.pdf"
    pdf.output(pdf_output)

    # Read the PDF file and send it as a response
    with open(pdf_output, 'rb') as f:
        pdf_data = f.read()

    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={name}_appointment.pdf'

    return response

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    # Extract form data
    age = request.form.get('age')
    blood_pressure = request.form.get('blood_pressure')
    specific_gravity = request.form.get('specific_gravity')
    albumin = request.form.get('albumin')
    sugar = request.form.get('sugar')
    red_blood_cells = request.form.get('red_blood_cells')
    pus_cell = request.form.get('pus_cell')
    pus_cell_clumps = request.form.get('pus_cell_clumps')
    bacteria = request.form.get('bacteria')
    blood_glucose_random = request.form.get('blood_glucose_random')
    blood_urea = request.form.get('blood_urea')
    serum_creatinine = request.form.get('serum_creatinine')
    sodium = request.form.get('sodium')
    potassium = request.form.get('potassium')
    haemoglobin = request.form.get('haemoglobin')
    packed_cell_volume = request.form.get('packed_cell_volume')
    white_blood_cell_count = request.form.get('white_blood_cell_count')
    red_blood_cell_count = request.form.get('red_blood_cell_count')
    hypertension = request.form.get('hypertension')
    diabetes_mellitus = request.form.get('diabetes_mellitus')
    coronary_artery_disease = request.form.get('coronary_artery_disease')
    appetite = request.form.get('appetite')
    peda_edema = request.form.get('peda_edema')
    aanemia = request.form.get('aanemia')

    
    user_input = [
        float(age), float(blood_pressure), float(specific_gravity), float(albumin), float(sugar),
        float(red_blood_cells), float(pus_cell), float(pus_cell_clumps), float(bacteria),
        float(blood_glucose_random), float(blood_urea), float(serum_creatinine), float(sodium),
        float(potassium), float(haemoglobin), float(packed_cell_volume), float(white_blood_cell_count),
        float(red_blood_cell_count), float(hypertension), float(diabetes_mellitus), float(coronary_artery_disease),
        float(appetite), float(peda_edema), float(aanemia)
    ]
    

    prediction = kidney_disease_model.predict([user_input])
    result = "The person has Kidney's disease" if prediction[0] == 1 else "The person does not have Kidney's disease"
    
    return render_template('result.html', result=result)

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    
    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')

    
    user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
    
 
    prediction = heart_disease_model.predict([user_input])
    result = "This person is having heart disease" if prediction[0] == 1 else "This person does not have any heart disease"
    
    return render_template('result.html', result=result)

# @app.route('/predict_diabetes', methods=['POST'])
# def predict_diabetes():
#     # Extract form data
#     pregnancies = request.form.get('pregnancies')
#     glucose = request.form.get('glucose')
#     # ...extract other fields...
#     user_input = [float(pregnancies), float(glucose), ...]
#     prediction = diabetes_model.predict([user_input])
#     result = "The person has diabetic" if prediction[0] == 1 else "The person has no diabetic"
#     return render_template('result.html', result=result)

@app.route('/get_detailed_info', methods=['POST'])
def get_detailed_info():
    data = request.json
    model = data['model']
    prediction = data['prediction']
    
    # Find the disease with the highest confidence score
    highest_confidence = max(prediction, key=lambda x: x['score'])
    disease = highest_confidence['label']
    
    # Prepare the payload for the Gemini API
    payload = {
        'disease': disease,
        'model': model
    }

    api_key = 'AIzaSyB-CJ12ikMfhhC3vguN00-mQInIGWw7Z5E'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post('https://diseaseinfo.onrender.com', json=payload, headers=headers)
    
    if response.status_code == 200:
        details = response.json().get('details', 'No details available.')
    else:
        details = 'Failed to fetch details from the Gemini API.'
    
    return jsonify({'details': details})

@app.route('/get_explanation', methods=['GET'])
def get_explanation():
    disease = request.args.get('disease')
    
    if not disease:
        return jsonify({'explanation': 'Disease not specified.'}), 400
    
    payload = {
        'disease': disease
    }

    try:

        response = requests.post('http://127.0.0.1:5002/v1/disease_info', json=payload)
        response.raise_for_status()


        data = response.json()
        explanation = data.get('info', 'No explanation available.')

        return jsonify({'explanation': explanation})

    except requests.exceptions.RequestException as e:
        return jsonify({'explanation': f"Failed to fetch explanation from the Gemini API: {e}"}), 500
    except Exception as e:
        return jsonify({'explanation': f"An unexpected error occurred: {e}"}), 500

@app.route('/upload_prescription', methods=['GET', 'POST'])
def upload_prescription():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']

        if file.filename == '':
            return 'No selected file', 400

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            
            image = Image.open(filepath)
            extracted_text = ocr_pipeline(image)

            
            os.remove(filepath)

            
            raw_text = extracted_text[0]['generated_text']

            
            text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prescription.txt')
            with open(text_file_path, 'w') as text_file:
                text_file.write(raw_text)

           
            return render_template('prescription_result.html', extracted_text=raw_text, text_file_path=text_file_path)

    return render_template('upload_prescription.html')

@app.route('/download_prescription')
def download_prescription():
    text_file_path = request.args.get('file_path')
    if text_file_path and os.path.exists(text_file_path):
        return send_file(text_file_path, as_attachment=True, download_name='prescription.txt')
    return 'File not found', 404


def init_db():
    conn = sqlite3.connect('portalauthentication.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            unique_id TEXT NOT NULL UNIQUE
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/authenticate', methods=['POST'])
def authenticate():
    data = request.json
    user_name = data.get('userName')
    user_id = data.get('userId')

    conn = sqlite3.connect('portalauthentication.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE name = ? AND unique_id = ?', (user_name, user_id))
    user = cursor.fetchone()
    conn.close()

    if user:
        return jsonify({'success': True, 'redirect_url': 'http://127.0.0.1:8000/'})
    else:
        return jsonify({'success': False, 'message': 'You are not authorized to access the portal'})


@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate_page():
    if request.method == 'POST':
        data = request.json
        user_name = data.get('userName')
        user_id = data.get('userId')

        conn = sqlite3.connect('portalauthentication.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE name = ? AND unique_id = ?', (user_name, user_id))
        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({'success': True, 'redirect_url': 'http://127.0.0.1:8000/'})
        else:
            return jsonify({'success': False, 'message': 'You are not authorized to access the portal'})
    return render_template('authenticate.html')

@app.route('/emergency', methods=['GET', 'POST'])
def emergency():
    if request.method == 'POST':
        category = request.form.get('category')
        file = request.files.get('file')
        user_location = request.form.get('location')  # Assume user provides location as "latitude,longitude"

        if not category or not user_location:
            return 'Category and location are required', 400

        # Validate user_location format
        try:
            latitude, longitude = map(str.strip, user_location.split(','))
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return 'Invalid location format. Please provide latitude and longitude separated by a comma (e.g., 17.3850,78.4867).', 400

        # Save uploaded image
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
        else:
            filepath = None

        # Find nearby hospitals using OpenStreetMap Nominatim API
        hospitals = find_nearby_hospitals(latitude, longitude)
        if not hospitals:
            return 'No nearby hospitals found', 404

        nearest_hospital = hospitals[0]
        hospital_name = nearest_hospital['display_name']
        hospital_address = nearest_hospital.get('address', {}).get('road', 'Address not available')

        # Send notification to the hospital
        send_notification(hospital_name, hospital_address, category, filepath)

        return render_template('emergency_success.html', hospital_name=hospital_name, hospital_address=hospital_address, latitude=latitude, longitude=longitude)

    return render_template('emergency.html')

def find_nearby_hospitals(latitude, longitude):
    """Find nearby hospitals using OpenStreetMap Nominatim API."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': 'hospital',
        'format': 'json',
        'addressdetails': 1,
        'limit': 5,
        'lat': latitude,
        'lon': longitude
    }
    headers = {
        'User-Agent': 'JeevaAI/1.0 (your_email@example.com)'  # Replace with your email
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching hospitals: {response.status_code}, {response.text}")
        return []

def send_notification(hospital_name, hospital_address, category, filepath):
    # Simulate sending an email notification to the hospital
    sender_email = "balajinbtt@gmail.com"
    receiver_email = "navuluribalaji03@gmail.com"
    subject = f"Emergency Alert: {category}"

    body = f"""
    An emergency case has been reported in the category: {category}.
    Please prepare necessary equipment and dispatch an ambulance to the location.
    Hospital: {hospital_name}
    Address: {hospital_address}
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    if filepath:
        with open(filepath, 'rb') as attachment:
            part = MIMEText(attachment.read(), 'base64', 'utf-8')
            part.add_header('Content-Disposition', f'attachment; filename={os.path. basename(filepath)}')
            msg.attach(part)

    # Send email (configure SMTP server)
    try:
        with smtplib.SMTP('smtp.example.com', 587) as server:
            server.starttls()
            server.login(sender_email, 'your_email_password')
            server.sendmail(sender_email, receiver_email, msg.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5001)