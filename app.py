from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import random
import pandas as pd
import mysql.connector
from datetime import datetime
from urllib.parse import unquote
# from google.cloud import translate_v2clear



app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Initialize Google Translate client
try:
    translate_client = translate_v2.Client()
except:
    translate_client = None

def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",  # Replace with your actual MySQL password
        database="skin"
    )

# Load model
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Doctor data
doctors = [
    {
        "name": "Dr. Sushruth Kamoji",
        "clinic": "Sparsh Skin Hair & Laser Clinic",
        "address": "Belagavi",
        "phone": "9876543210",
        "jio_location": "https://maps.app.goo.gl/BeFtnEz5tryd53ob8"
    },
    {
        "name": "Dr. Shivakumar Patil",
        "clinic": "Ashwini Medical Center",
        "address": "Belagavi",
        "phone": "9845123456",
        "jio_location": "https://maps.app.goo.gl/XmME8EXePiTwAzFY6"
    },
    {
        "name": "Dr. Kutre",
        "clinic": "Kutre’s Skin Care Centre",
        "address": "Belagavi",
        "phone":"9354563278",
        "jio_location": "https://maps.app.goo.gl/YqQKb9yQJN1TJmxt6"
    },
    {
        "name": "Dr. A M Pandit",
        "clinic": "Skin care centre",
        "address": "Belagavi",
        "phone": "9448133475",
        "jio_location": "https://www.google.com/maps/dir//1951%2F1,+Kadolkar+Galli,+Raviwar+Peth"
    },
    {
        "name": "Dr. Deepa Gunji",
        "clinic": "Skinfinite",
        "address": "Belagavi",
        "phone": "9888877665",
        "jio_location": "https://maps.app.goo.gl/XffnMytNsYGR4F2D6"
    },
    {
    "name": "Dr. Mayuri Sawant",
    "clinic": "Mayur Clinic",
    "address": "Belagavi",
    "phone": "9797979797",
    "jio_location": "https://www.google.com/maps/search/?api=1&query=Mayur+Clinic+Belagavi"
},

    {
        "name": "Dr. Santosh Shinde",
        "clinic": "Cutis Clinic",
        "address": "Belagavi",
        "phone": "9786453210",
        "jio_location": "https://maps.app.goo.gl/PWPKxQqwtSyarRXW7"
    },
    {
        "name": "Dr. Sharada Goudagaon",
        "clinic": "Patil Clinic",
        "address": "Belagavi",
        "phone": "9654321870",
        "jio_location": "https://maps.app.goo.gl/npLGXYAHyxZ2xQrM9"
    },
    {
        "name": "Dr. Gajanan Pise",
        "clinic": "Mangal Clinic",
        "address": "Belagavi",
        "phone": "9765432180",
        "jio_location": "https://maps.app.goo.gl/nnbmM1aJrDDRsB5e6"
    }
]

@app.route('/')
def root():
    return redirect('/login')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash("Please login to continue.")
        return redirect('/login')

    # ✅ Safely get values from session
    class_name = session.get('class_name')
    confidence_score = session.get('confidence_score')
    file_path = session.get('uploaded_image_path')

    return render_template('index.html',
                           class_name=class_name,
                           confidence_score=confidence_score,
                           file_path=file_path,
                           show_patient_form=True if class_name else False,
                           prediction_done=True if class_name else False)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash("Please upload an image.")
        return redirect('/dashboard')

    file = request.files['file']
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        session['uploaded_image_path'] = f"/{file_path}"

        image = Image.open(file_path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip().split(" ", 1)[1]  # ✅ correct
        confidence_score = round(prediction[0][index] * 100, 2)

    # ✅ Store in session
        session['image_uploaded'] = True
        session['prediction_done'] = True
        session['class_name'] = class_name
        session['confidence_score'] = confidence_score

        return redirect('/dashboard')

    except Exception as e:
        flash(f"Error during prediction: {e}")
        return redirect('/dashboard')


@app.route('/save_patient', methods=['POST'])
def save_patient():
    name = request.form['name']
    address = request.form['address']
    contact = request.form['contact']
    prediction = request.form['prediction']
    confidence = request.form['confidence']
    date = datetime.now().strftime("%Y-%m-%d")

    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients (date, name, address, contact, prediction, confidence)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (date, name, address, contact, prediction, confidence))
    conn.commit()
    conn.close()

    flash("Patient record saved.")
    return redirect('/dashboard')


@app.route('/show_severity')
def show_severity():
    if not session.get('image_uploaded'):
        flash('Please upload an image first.', 'info')
        return redirect('/dashboard')

    file_path = session.get('uploaded_image_path')
    class_name = session.get('class_name')
    confidence_score = session.get('confidence_score')

    if not class_name:
        flash('Prediction not available.')
        return redirect('/dashboard')

    # 🔧 Clean the class name to avoid extra spaces/newlines
    class_name = class_name.strip()

    # 🔑 Correct mapping based on your class names
    severity_mapping = {
        "AtopiDermatitis": "High",
        "BasalCellCarcinoma": "High",
        "BenignKeratosis": "Low",
        "Eczema": "High",
        "MelanocyticNevi": "Medium",
        "Melanoma": "High",
        "Psoriasis": "Medium",
        "SeborrheicKeratoses": "Low",
        "TineaRingwormCandidiasis": "Low",
        "WartsMolluscum": "Low"
    }

    stage = severity_mapping.get(class_name)

    # ✅ Warning shown only for High severity
    show_warning = (stage == "High")

    return render_template('index.html',
                           prediction_result=False,
                           stage=stage,
                           show_warning=show_warning,
                           file_path=file_path,
                           class_name=class_name)



    # 🔧 Solutions dictionary based on predicted disease
@app.route('/show_solutions')
def show_solutions():
    if not session.get('image_uploaded'):
        flash('Please upload an image first.', 'info')
        return redirect('/dashboard')

    file_path = session.get('uploaded_image_path')
    class_name = session.get('class_name', '').strip()

    solutions = {
        "AtopiDermatitis": {
            "diet": "Avoid common allergens like dairy, nuts, and eggs. Drink more water.",
            "medicine": "Levocetirizine 5mg at night",
            "creams": [{"name": "Efatop PE", "brand": "Glenmark", "usage": "Apply after bath"}],
            "lotions": [{"name": "Venusia Max Lotion", "brand": "Dr. Reddy's", "usage": "Twice daily"}],
            "soaps": [{"name": "Cetaphil Cleansing Bar", "brand": "Galderma", "usage": "Use for bathing"}]
        },
        "BasalCellCarcinoma": {
            "diet": "Include antioxidants (green leafy veggies, tomatoes), avoid direct sunlight.",
            "medicine": "Imiquad Cream (only under medical supervision)",
            "creams": [{"name": "Imiquad", "brand": "Glenmark", "usage": "Use as prescribed by oncologist"}],
            "lotions": [{"name": "Aloe Vera Gel", "brand": "Patanjali", "usage": "Apply as soothing agent"}],
            "soaps": [{"name": "Dove Sensitive", "brand": "Unilever India", "usage": "Gentle cleansing"}]
        },
        "BenignKeratosis": {
            "diet": "Balanced diet, stay hydrated. No major restrictions.",
            "medicine": "No routine medicine needed unless advised",
            "creams": [{"name": "Moisturex Soft Cream", "brand": "Cipla", "usage": "Use for dry areas"}],
            "lotions": [{"name": "Lacsoft Lotion", "brand": "Ipca", "usage": "Apply once or twice daily"}],
            "soaps": [{"name": "Syndet Bar", "brand": "Sebamed", "usage": "Use instead of regular soap"}]
        },
        "Eczema": {
            "diet": "Avoid spicy, acidic food and dairy products.",
            "medicine": "Atarax 10mg once daily",
            "creams": [{"name": "Candiderm Cream", "brand": "Glenmark", "usage": "Apply thin layer"}],
            "lotions": [{"name": "Emodel Lotion", "brand": "Zydus", "usage": "Apply after bath"}],
            "soaps": [{"name": "Oilatum Soap", "brand": "Stiefel India", "usage": "Use instead of regular soap"}]
        },
        "MelanocyticNevi": {
            "diet": "Eat antioxidant-rich foods, avoid sun exposure.",
            "medicine": "Regular monitoring; no treatment unless changes occur",
            "creams": [{"name": "Photoban 50 Sunscreen", "brand": "Ranbaxy", "usage": "Apply before sun exposure"}],
            "lotions": [{"name": "Aloe Vera Gel", "brand": "Patanjali", "usage": "Cooling effect"}],
            "soaps": [{"name": "Sebamed Baby Soap", "brand": "Sebamed", "usage": "Use gentle soap daily"}]
        },
        "Melanoma": {
            "diet": "Avoid processed meats, increase vegetables and Vitamin D intake.",
            "medicine": "Oncology consultation required for targeted therapy",
            "creams": [{"name": "As prescribed by doctor", "brand": "-", "usage": "Do not self-medicate"}],
            "lotions": [{"name": "Calosoft AF Lotion", "brand": "Glenmark", "usage": "For comfort only"}],
            "soaps": [{"name": "Syndet Cleansing Bar", "brand": "Curatio", "usage": "Mild cleansing only"}]
        },
        "Psoriasis": {
            "diet": "Avoid red meat, alcohol. Increase fish oil and green leafy vegetables.",
            "medicine": "Betnovate Cream (Betamethasone) — apply with care",
            "creams": [{"name": "Tacroz Forte", "brand": "Glenmark", "usage": "Apply to thick plaques"}],
            "lotions": [{"name": "Salytar Lotion", "brand": "Ajanta", "usage": "Apply twice daily"}],
            "soaps": [{"name": "Coal Tar Soap", "brand": "Psorolin by Dr. JRK’s", "usage": "Use during bath"}]
        },
        "SeborrheicKeratoses": {
            "diet": "No strict diet. Maintain good skin hygiene.",
            "medicine": "Consult dermatologist if lesion changes",
            "creams": [{"name": "U-Lactin Cream", "brand": "Eris Lifesciences", "usage": "Use once daily"}],
            "lotions": [{"name": "Moisturex Lotion", "brand": "Cipla", "usage": "Hydration and softness"}],
            "soaps": [{"name": "Dermosafe Soap", "brand": "Curatio", "usage": "Daily use recommended"}]
        },
        "TineaRingwormCandidiasis": {
            "diet": "Avoid sugar, wear breathable clothing, avoid dampness.",
            "medicine": "Fluconazole 150mg weekly (consult doctor)",
            "creams": [{"name": "Clocip Cream", "brand": "Cipla", "usage": "Twice daily on infected area"}],
            "lotions": [{"name": "Lulifin Lotion", "brand": "Sun Pharma", "usage": "Apply thin layer"}],
            "soaps": [{"name": "Ketocip Soap", "brand": "Cipla", "usage": "Use daily during treatment"}]
        },
       "WartsMolluscum": {
            "diet": "Boost immunity with citrus fruits, tulsi, and turmeric milk.",
            "medicine": "Salicylic acid topical (consult doctor before use)",
            "creams": [{"name": "DuoFilm", "brand": "Menarini", "usage": "Apply only on wart"}],
            "lotions": [{"name": "Immuzest Lotion", "brand": "Meyer", "usage": "To support immune skin health"}],
            "soaps": [{"name": "Neem Soap", "brand": "Himalaya", "usage": "Twice daily for hygiene"}]
        }
    }

    default_solution = {
        "diet": "Maintain general hygiene and eat a balanced diet.",
        "medicine": "Consult a dermatologist for personalized treatment.",
        "creams": [],
        "lotions": [],
        "soaps": []
    }

    solution = solutions.get(class_name, default_solution)

    return render_template('index.html',
                           diet=solution['diet'],
                           medicine=solution['medicine'],
                           creams=solution['creams'],
                           lotions=solution['lotions'],
                           soaps=solution['soaps'],
                           file_path=file_path,
                           prediction_done=False)



@app.route('/suggest_doctor')
def suggest_doctor():
    if not session.get('image_uploaded'):
        flash('Please upload an image first.', 'info')
        return redirect('/dashboard')

    file_path = session.get('uploaded_image_path')  # this was set in /predict
    return render_template('index.html',
                           selected_doctors=random.sample(doctors, 2),
                           file_path=file_path,
                           prediction_done=False)

@app.route('/take_test')
def take_test():
    if not session.get('image_uploaded'):
        flash('Please upload an image first.', 'info')
        return redirect('/dashboard')

    return render_template('index.html', show_take_test=True, prediction_done=False)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            flash("User already exists.", "error")
            conn.close()
            return redirect('/register')

        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                       (username, email, password))
        conn.commit()
        conn.close()

        flash("Registered successfully!", "success")
        return render_template('register.html')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if email == "admin@gmail.com" and password == "admin":
            session['admin'] = True
            return redirect('/admin_dashboard')

        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user'] = email
            return redirect('/dashboard')
        else:
            flash("Invalid credentials")
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        new_password = request.form['new_password']

        conn = get_mysql_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = %s", (email,))
        if c.fetchone():
            c.execute("UPDATE users SET password = %s WHERE email = %s", (new_password, email))
            conn.commit()
            conn.close()
            flash("Password updated successfully!")
        else:
            flash("Email not found.")
        return redirect('/forgot_password')
    return render_template('forgot_password.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin' not in session:
        flash("Access denied.")
        return redirect('/login')

    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patients")
    prediction_count = cursor.fetchone()[0]
    conn.close()

    return render_template("admin_dashboard.html",
                           title="Admin Dashboard",
                           user_count=user_count,
                           prediction_count=prediction_count)


@app.route('/show_patients')
def show_patients():
    if 'admin' not in session:
        flash("Access denied.")
        return redirect('/login')

    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT date, name, address, contact, prediction, confidence FROM patients")
    rows = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=["Date", "Name", "Address", "Contact", "Prediction", "Confidence"])
    html_table = df.to_html(index=False)

    return render_template("admin_dashboard.html",
                           title="Patient Records",
                           table=html_table)

@app.route('/show_users')
def show_users():
    if 'admin' not in session:
        flash("Access denied.")
        return redirect('/login')

    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, email, password FROM users")
    rows = cursor.fetchall()
    users = [{"Username": r[0], "Email": r[1], "Password": r[2]} for r in rows]
    conn.close()

    return render_template("admin_dashboard.html",
                           title="Registered Users",
                           users=users)

@app.route('/edit_user/<email>', methods=['GET', 'POST'])
def edit_user(email):
    conn = get_mysql_connection()
    cursor = conn.cursor()

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor.execute("UPDATE users SET username = %s, password = %s WHERE email = %s",
                       (username, password, email))
        conn.commit()
        conn.close()
        flash("User updated successfully.")
        return redirect('/show_users')

    # Fetch user info for GET request
    cursor.execute("SELECT username, password FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return render_template("edit_user.html",
                               email=email,
                               username=user[0],
                               password=user[1])
    else:
        flash("User not found.")
        return redirect('/show_users')

@app.route('/delete_user/<path:email>')
def delete_user(email):
    if 'admin' not in session:
        flash("Access denied.")
        return redirect('/login')

    email = unquote(email)

    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE LOWER(email) = %s", (email.lower(),))
        conn.commit()
        conn.close()
        flash("User deleted successfully.")
    except Exception as e:
        flash(f"Error deleting user: {e}")

    return redirect('/show_users')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.")
    return redirect('/login')

@app.route("/test_analysis", methods=["POST"])
def test_analysis():
    q1 = request.form.get("q1")
    q2 = request.form.get("q2")
    q3 = request.form.get("q3")
    q4 = request.form.get("q4")
    q5 = request.form.get("q5")

    score = sum([q1 == "Yes", q2 == "Yes", q3 == "Yes", q4 == "Yes", q5 == "Yes"])

    # 🟢 No symptoms selected
    if score == 0:
        class_name = "No significant skin issue detected"
        recommended_medicine = None
        recommended_soap = None
        recommended_doctor = None

    elif score >= 4:
        class_name = "Eczema"
        recommended_medicine = "Cetirizine"
        recommended_soap = "Dove Sensitive"
        recommended_doctor = "Dr. Skin Specialist"

    elif score == 3:
        class_name = "Psoriasis"
        recommended_medicine = "Hydrocortisone"
        recommended_soap = "Cetaphil"
        recommended_doctor = "Dr. Psoriasis Expert"

    else:
        class_name = "Fungal Infection"
        recommended_medicine = "Fluconazole"
        recommended_soap = "Nizoral"
        recommended_doctor = "Dr. Fungal Specialist"

    return render_template("index.html", 
        test_class_name=class_name,
        recommended_medicine=recommended_medicine,
        recommended_soap=recommended_soap,
        recommended_doctor=recommended_doctor
    )

@app.route('/translate', methods=['POST'])
def translate():
    """Translate text to specified language"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('target_language', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not translate_client:
            # Fallback: return original text if translator not available
            return jsonify({'translatedText': text, 'status': 'no_translator'})
        
        # Translate text using Google Translate API
        result = translate_client.translate_text(text, target_language=target_language)
        translated_text = result.get('translatedText', text)
        
        return jsonify({
            'translatedText': translated_text,
            'status': 'success',
            'source_language': result.get('detectedSourceLanguage', 'auto')
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
