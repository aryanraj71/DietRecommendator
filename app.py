from flask import Flask, render_template, request, jsonify, make_response, send_file, session, redirect, url_for
from fpdf import FPDF #type: ignore
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import tempfile
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient #type: ignore
from bson.objectid import ObjectId #type: ignore
import secrets
import datetime

app = Flask(__name__, static_url_path='/static')
app.secret_key = secrets.token_hex(16)  

# MongoDB Configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['diet_recommendator']
users_collection = db['users']
reports_collection = db['reports']  

try:
    data = pd.read_csv('food.csv')
    required_columns = ['Food_items', 'Weight_Gain', 'Healthy', 'Weight_Loss',
                       'Vegetarian', 'Non_Vegetarian', 'Diabetes_Friendly',
                       'Low_Sodium', 'Gluten_Free', 'Lactose_Free',
                       'Nut_Free', 'Meal_Type', 'Calories', 'Protein', 'Carbs', 'Fat']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    nutritional_data = data[['Calories', 'Protein', 'Carbs', 'Fat']].dropna()
    scaler = StandardScaler()
    nutritional_scaled = scaler.fit_transform(nutritional_data)
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['Nutrition_Cluster'] = kmeans.fit_predict(nutritional_scaled)
    pca = PCA(n_components=2)
    nutritional_pca = pca.fit_transform(nutritional_scaled)
    data['PCA1'] = nutritional_pca[:, 0]
    data['PCA2'] = nutritional_pca[:, 1]

    calorie_bins = pd.qcut(data['Calories'].dropna(), q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(nutritional_data, calorie_bins, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    data['Calorie_Level'] = rf.predict(nutritional_data)
    calorie_quantiles = data['Calories'].quantile([0.2, 0.4, 0.6, 0.8]).values
    LOW_CALORIE_THRESHOLD_HIGH = calorie_quantiles[1]
    AVG_CALORIE_THRESHOLD_LOW = calorie_quantiles[0]
    AVG_CALORIE_THRESHOLD_HIGH = calorie_quantiles[3]
    HIGH_CALORIE_THRESHOLD_LOW = calorie_quantiles[2]
    MIN_OPTIONS = 8

except Exception as e:
    print(f"Error loading data: {e}")
    data = pd.DataFrame()

# Auth Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.json.get('email')
            password = request.json.get('password')
            user = users_collection.find_one({'email': email})
            
            if user and check_password_hash(user['password'], password):
                session['user_id'] = str(user['_id'])
                session['user_email'] = user['email']
                return jsonify({'success': True})
            return jsonify({'error': 'Invalid credentials'}), 401
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            user_data = request.json
            if users_collection.find_one({'email': user_data['email']}):
                return jsonify({'error': 'Email already exists'}), 400
            
            users_collection.insert_one({
                'name': user_data['name'],
                'email': user_data['email'],
                'password': generate_password_hash(user_data['password']),
                'created_at': datetime.datetime.utcnow()
            })
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# Protected Routes
@app.route('/get-started')
def get_started():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def classify_bmi(bmi):
    if bmi < 16:
        return 4, "Severely Underweight"
    elif 16 <= bmi < 18.5:
        return 3, "Underweight"
    elif 18.5 <= bmi < 25:
        return 2, "Healthy"
    elif 25 <= bmi < 30:
        return 1, "Overweight"
    else:
        return 0, "Severely Overweight"

def predict_current_nutrients(bmi, diet_type):
    """Predict current nutrient intake with comprehensive BMI-based adjustments."""
    bmi_category, _ = classify_bmi(bmi)
    
    # Base values for a healthy individual
    base_values = {
        'Calories': 2500,
        'Protein': 60,
        'Fat': 65,
        'Carbs': 300,
        'Iron': 18,
        'Calcium': 1000,
        'Sodium': 2300,
        'Potassium': 3400,
        'Fiber': 25,
        'VitaminD': 15,
        'Sugars': 50
    }

    # Multipliers based on BMI severity (applies to all nutrients)
    if bmi_category == 4:  # Severely Underweight
        multipliers = {
            'Calories': 0.6,  
            'Protein': 0.8,
            'Fat': 0.85,
            'Carbs': 0.9,
            'Micronutrients': 0.9  
        }
    elif bmi_category == 3:  
        multipliers = {
            'Calories': 0.8,  
            'Protein': 0.9,
            'Fat': 0.9,
            'Carbs': 0.95,
            'Micronutrients': 0.95
        }
    elif bmi_category == 1: 
        multipliers = {
            'Calories': 1.2,  
            'Protein': 0.95,
            'Fat': 1.1,
            'Carbs': 1.05,
            'Micronutrients': 1.05
        }
    elif bmi_category == 0: 
        multipliers = {
            'Calories': 1.4,  
            'Protein': 0.9,
            'Fat': 1.2,
            'Carbs': 1.1,
            'Micronutrients': 1.1
        }
    else:  
        multipliers = {
            'Calories': 1.0,
            'Protein': 1.0,
            'Fat': 1.0,
            'Carbs': 1.0,
            'Micronutrients': 1.0
        }

    current = {
        'Calories': round(base_values['Calories'] * multipliers['Calories']),
        'Protein': round(base_values['Protein'] * multipliers['Protein']),
        'Fat': round(base_values['Fat'] * multipliers['Fat']),
        'Carbs': round(base_values['Carbs'] * multipliers['Carbs']),
        'Iron': round(base_values['Iron'] * multipliers['Micronutrients']),
        'Calcium': round(base_values['Calcium'] * multipliers['Micronutrients']),
        'Sodium': round(base_values['Sodium'] * multipliers['Micronutrients']),
        'Potassium': round(base_values['Potassium'] * multipliers['Micronutrients']),
        'Fiber': round(base_values['Fiber'] * multipliers['Micronutrients']),
        'VitaminD': round(base_values['VitaminD'] * multipliers['Micronutrients']),
        'Sugars': round(base_values['Sugars'] * multipliers['Micronutrients'])
    }

    # Additional diet-type adjustments
    if diet_type == "vegetarian":
        current['Protein'] = round(current['Protein'] * 0.9)
        current['Iron'] = round(current['Iron'] * 1.1)  
    elif diet_type == "non-vegetarian":
        current['Protein'] = round(current['Protein'] * 1.1)

    return current

def get_target_nutrients(bmi, diet_type):
    """Set comprehensive targets based on BMI and diet type."""
    bmi_category, _ = classify_bmi(bmi)
    
    # Base targets for healthy individual
    targets = {
        'Calories': 2500,
        'Protein': 60,
        'Fat': 65,
        'Carbs': 300,
        'Iron': 18,
        'Calcium': 1000,
        'Sodium': 2300,
        'Potassium': 3400,
        'Fiber': 25,
        'VitaminD': 15,
        'Sugars': 50
    }

    # BMI-based adjustments
    if bmi_category == 4:
        adjustments = {
            'Calories': 1.4,     
            'Protein': 1.25,
            'Fat': 1.15,
            'Carbs': 1.1,
            'Micronutrients': 1.15
        }
    elif bmi_category == 3:  
        adjustments = {
            'Calories': 1.2,  
            'Protein': 1.15,
            'Fat': 1.1,
            'Carbs': 1.05,
            'Micronutrients': 1.05
        }
    elif bmi_category == 1:  
        adjustments = {
            'Calories': 0.8,  
            'Protein': 0.95,
            'Fat': 0.85,
            'Carbs': 0.9,
            'Micronutrients': 0.95
        }
    elif bmi_category == 0:  
        adjustments = {
            'Calories': 0.6,  
            'Protein': 0.9,
            'Fat': 0.7,
            'Carbs': 0.85,
            'Micronutrients': 0.9
        }
    else:  # Healthy
        adjustments = {
            'Calories': 1.0,
            'Protein': 1.0,
            'Fat': 1.0,
            'Carbs': 1.0,
            'Micronutrients': 1.0
        }

    # Apply adjustments
    targets = {
        'Calories': round(targets['Calories'] * adjustments['Calories']),
        'Protein': round(targets['Protein'] * adjustments['Protein']),
        'Fat': round(targets['Fat'] * adjustments['Fat']),
        'Carbs': round(targets['Carbs'] * adjustments['Carbs']),
        'Iron': round(targets['Iron'] * adjustments['Micronutrients']),
        'Calcium': round(targets['Calcium'] * adjustments['Micronutrients']),
        'Sodium': round(targets['Sodium'] * adjustments['Micronutrients']),
        'Potassium': round(targets['Potassium'] * adjustments['Micronutrients']),
        'Fiber': round(targets['Fiber'] * adjustments['Micronutrients']),
        'VitaminD': round(targets['VitaminD'] * adjustments['Micronutrients']),
        'Sugars': round(targets['Sugars'] * adjustments['Micronutrients'])
    }

    # Diet-specific fine-tuning
    if diet_type == "vegetarian":
        targets.update({
            'Protein': round(targets['Protein'] * 0.9),
            'Iron': round(targets['Iron'] * 1.2)  
        })
    elif diet_type == "non-vegetarian":
        targets['Protein'] = round(targets['Protein'] * 1.1)

    return targets

def generate_nutrient_comparison_plot(current, target):
    """Generate a bar chart comparing current vs target intake."""
    nutrients = list(current.keys())
    current_values = list(current.values())
    target_values = [target[n] for n in nutrients]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(nutrients))

    plt.bar(index, current_values, bar_width, color='#1f77b4', label='Current Intake')
    plt.bar(index + bar_width, target_values, bar_width, color='#ff7f0e', label='Target Intake')

    plt.xlabel('Nutrients')
    plt.ylabel('Amount')
    plt.title('Current vs Target Nutrient Intake')
    plt.xticks(index + bar_width/2, nutrients, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('ascii')
    plt.close()

    return plot_data

def generate_comparison_plot(previous, current, title="Nutrient Comparison"):
    """Generate a bar chart comparing previous vs current report."""
    nutrients = ['Calories', 'Protein', 'Carbs', 'Fat']
    previous_values = [previous['nutrients'].get(nutrient, 0) for nutrient in nutrients]
    current_values = [current['nutrients'].get(nutrient, 0) for nutrient in nutrients]

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(nutrients))

    plt.bar(index, previous_values, bar_width, color='#1f77b4', label='Previous Report')
    plt.bar(index + bar_width, current_values, bar_width, color='#ff7f0e', label='Current Report')

    plt.xlabel('Nutrients')
    plt.ylabel('Amount')
    plt.title(title)
    plt.xticks(index + bar_width/2, nutrients)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('ascii')
    plt.close()

    return plot_data

def get_food_options(bmi_category, meal_type, diet_type, restrictions):
    if data.empty:
        print("Warning: Data is empty in get_food_options")
        return []

    filtered = data[data['Meal_Type'] == meal_type].copy()
    print(f"Initial filtered for {meal_type}: {len(filtered)} items")

    if diet_type == "vegetarian":
        filtered = filtered[filtered['Vegetarian'] == 1]
        print(f"After vegetarian filter for {meal_type}: {len(filtered)} items")
    elif diet_type == "non-vegetarian":
        filtered = filtered[filtered['Non_Vegetarian'] == 1]
        print(f"After non-vegetarian filter for {meal_type}: {len(filtered)} items")
    else:  # "both"
        print(f"Using 'both' diet type for {meal_type}: {len(filtered)} items")

    if restrictions:
        print(f"Applying restrictions: {restrictions}")
        if 'diabetes' in restrictions:
            filtered = filtered[filtered['Diabetes_Friendly'] == 1]
        if 'hypertension' in restrictions:
            filtered = filtered[filtered['Low_Sodium'] == 1]
        if 'celiac' in restrictions:
            filtered = filtered[filtered['Gluten_Free'] == 1]
        if 'lactose' in restrictions:
            filtered = filtered[filtered['Lactose_Free'] == 1]
        if 'nutAllergy' in restrictions:
            filtered = filtered[filtered['Nut_Free'] == 1]
        print(f"After restrictions for {meal_type}: {len(filtered)} items")

    if diet_type in ["vegetarian", "both"]:
        if bmi_category in [0, 1]:  
            filtered = filtered[filtered['Calories'] <= LOW_CALORIE_THRESHOLD_HIGH]
        elif bmi_category == 2: 
            filtered = filtered[(filtered['Calories'] > AVG_CALORIE_THRESHOLD_LOW) & 
                              (filtered['Calories'] <= AVG_CALORIE_THRESHOLD_HIGH)]
        elif bmi_category in [3, 4]:  
            filtered = filtered[filtered['Calories'] >= HIGH_CALORIE_THRESHOLD_LOW]
        print(f"After initial calorie filter for {meal_type} (BMI {bmi_category}): {len(filtered)} items")

        if len(filtered) < MIN_OPTIONS:
            print(f"Expanding options for {meal_type}: only {len(filtered)} items found")
            base_filtered = filtered.copy()
            all_options = data[data['Meal_Type'] == meal_type].copy()
            if diet_type == "vegetarian":
                all_options = all_options[all_options['Vegetarian'] == 1]
            elif diet_type == "both":
                pass
            if restrictions:
                if 'diabetes' in restrictions:
                    all_options = all_options[all_options['Diabetes_Friendly'] == 1]
                if 'hypertension' in restrictions:
                    all_options = all_options[all_options['Low_Sodium'] == 1]
                if 'celiac' in restrictions:
                    all_options = all_options[all_options['Gluten_Free'] == 1]
                if 'lactose' in restrictions:
                    all_options = all_options[all_options['Lactose_Free'] == 1]
                if 'nutAllergy' in restrictions:
                    all_options = all_options[all_options['Nut_Free'] == 1]

            unique_clusters = base_filtered['Nutrition_Cluster'].unique()
            additional_options = pd.DataFrame()
            for cluster in unique_clusters:
                cluster_options = all_options[all_options['Nutrition_Cluster'] == cluster]
                additional_options = pd.concat([additional_options, cluster_options])
                if len(pd.concat([base_filtered, additional_options]).drop_duplicates()) >= MIN_OPTIONS:
                    break
            filtered = pd.concat([base_filtered, additional_options]).drop_duplicates()[:MIN_OPTIONS]
            print(f"After expansion for {meal_type}: {len(filtered)} items")

    else:  
        print(f"No calorie filter applied for non-vegetarian {meal_type}: {len(filtered)} items")

    if not filtered.empty:
        filtered = filtered.sort_values(['Nutrition_Cluster', 'Calorie_Level'], 
                                      ascending=[True, bmi_category in [0, 1]])

    food_options = filtered[['Food_items', 'Calories', 'Protein', 'Carbs', 'Fat']].to_dict('records')
    print(f"Final food options for {meal_type}: {food_options}")
    return food_options

def generate_nutrition_plot(food_items):
    if data.empty or not food_items:
        print("Warning: No food items or data available for nutrition plot")
        return None

    plt.figure(figsize=(10, 6))
    recommended = data[data['Food_items'].isin(food_items)].dropna(subset=['Protein', 'Carbs', 'Fat'])

    if recommended.empty:
        print(f"Warning: No matching food items found for nutrition plot: {food_items}")
        plt.close()
        return None

    colors = {'Protein': 'blue', 'Carbs': 'green', 'Fat': 'orange'}
    
    for idx, row in recommended.iterrows():
        protein = row['Protein']
        carbs = row['Carbs']
        fat = row['Fat']
        food = row['Food_items']

        plt.bar(food, protein, color=colors['Protein'], label='Protein' if idx == 0 else "")
        plt.bar(food, carbs, bottom=protein, color=colors['Carbs'], label='Carbs' if idx == 0 else "")
        plt.bar(food, fat, bottom=protein + carbs, color=colors['Fat'], label='Fat' if idx == 0 else "")

    plt.title('Nutritional Composition of Selected Foods')
    plt.ylabel('Grams per serving')
    plt.xticks(rotation=45, ha='right')
    
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors['Protein'], label='Protein'),
        plt.Rectangle((0, 0), 1, 1, color=colors['Carbs'], label='Carbohydrates'),
        plt.Rectangle((0, 0), 1, 1, color=colors['Fat'], label='Fat')
    ]
    plt.legend(handles=handles, title='Nutrients', loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('ascii')
    plt.close()

    return plot_data

def generate_cluster_plot():
    if data.empty:
        print("Warning: No data available for cluster plot")
        return None

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data['PCA1'].dropna(), data['PCA2'].dropna(), c=data['Nutrition_Cluster'].dropna(),
                         cmap='viridis', alpha=0.6)
    plt.title('Food Clusters Based on Nutritional Composition')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('ascii')
    plt.close()

    return plot_data

def create_pdf(selected_foods, user_data, bmi, bmi_category, plot_data=None, cluster_plot=None, nutrient_plot=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Personal Information
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Personalized Meal Plan", ln=1, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Your Health Profile", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Name: {user_data.get('name', 'Not provided')}", ln=1)
    pdf.cell(200, 10, txt=f"Age: {user_data.get('age', 'Not provided')}", ln=1)
    pdf.cell(200, 10, txt=f"Gender: {user_data.get('gender', 'Not provided')}", ln=1)
    pdf.cell(200, 10, txt=f"BMI: {bmi:.1f} - {bmi_category}", ln=1)
    pdf.ln(15)

    # Cluster Plot
    if cluster_plot:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(base64.b64decode(cluster_plot))
                temp_file_path = temp_file.name
            pdf.image(temp_file_path, x=10, y=pdf.get_y(), w=180, h=100)
            pdf.ln(110)
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Failed to add cluster plot to PDF: {e}")

    # Nutrient Comparison Plot
    if nutrient_plot:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(base64.b64decode(nutrient_plot))
                temp_file_path = temp_file.name
            pdf.image(temp_file_path, x=10, y=pdf.get_y(), w=180, h=100)
            pdf.ln(110)
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Failed to add nutrient plot to PDF: {e}")

    # Meal Plan
    meal_order = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Your Selected Meals", ln=1)
    pdf.ln(5)

    for meal_type in meal_order:
        if meal_type in selected_foods and selected_foods[meal_type]:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt=meal_type, ln=1)
            pdf.set_font("Arial", size=10)
            for food in selected_foods[meal_type]:
                pdf.cell(200, 8, txt=f"- {food.get('name', 'Unknown')}", ln=1)
                pdf.cell(200, 8, txt=f"  Calories: {food.get('calories', 0)}, Protein: {food.get('protein', 0)}g, Carbs: {food.get('carbs', 0)}g, Fat: {food.get('fat', 0)}g", ln=1)
            pdf.ln(5)

    # Nutrition Plot
    if plot_data:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(base64.b64decode(plot_data))
                temp_file_path = temp_file.name
            if pdf.get_y() > 180:
                pdf.add_page()
            pdf.image(temp_file_path, x=10, y=pdf.get_y(), w=180, h=100)
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Failed to add nutrition plot to PDF: {e}")

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_pdf.name)
    temp_pdf.close()
    
    return temp_pdf.name

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/get-nutrient-analysis', methods=['POST'])
def get_nutrient_analysis():
    try:
        request_data = request.get_json()
        bmi = float(request_data['bmi'])
        diet_type = request_data['diet_type']

        current = predict_current_nutrients(bmi, diet_type)
        target = get_target_nutrients(bmi, diet_type)
        nutrient_plot = generate_nutrient_comparison_plot(current, target)

        return jsonify({
            "nutrient_plot": nutrient_plot,
            "current": current,
            "target": target
        })

    except Exception as e:
        print(f"Error in get-nutrient-analysis: {e}")
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

@app.route('/get-food-options', methods=['POST'])
def get_food_options_route():
    try:
        if data.empty:
            return jsonify({"error": "Food database not loaded properly"}), 500

        request_data = request.get_json()
        bmi_category = int(request_data['bmi_category'])
        meal_type = request_data['meal_type']
        diet_type = request_data['diet_type']
        restrictions = request_data.get('restrictions', [])

        food_options = get_food_options(bmi_category, meal_type, diet_type, restrictions)
        print(f"Debug: Food options for {meal_type}: {food_options}")
        
        return jsonify({
            "options": food_options
        })

    except Exception as e:
        print(f"Error in get-food-options: {e}")
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

@app.route('/generate-plan', methods=['POST'])
def generate_plan():
    try:
        if data.empty:
            return jsonify({"error": "Food database not loaded properly"}), 500

        request_data = request.get_json()
        selected_foods = request_data.get('selected_foods', {})
        user_data = request_data.get('user_data', {})
        bmi = float(request_data.get('bmi', 0))
        bmi_category = request_data.get('bmi_category', 'Unknown')
        diet_type = request_data.get('diet_type', 'both')

        # Validate selected_foods
        print(f"Debug: Received selected_foods: {selected_foods}")
        if not selected_foods or all(len(foods) == 0 for foods in selected_foods.values()):
            print("Warning: No food items selected")
            return jsonify({"error": "Please select at least one food item"}), 400

        food_names = [food['name'] for meal in selected_foods.values() for food in meal if 'name' in food]
        print(f"Debug: Extracted food names: {food_names}")
        nutrition_plot = generate_nutrition_plot(food_names)
        cluster_plot = generate_cluster_plot()

        # Calculate total nutrients for the selected meal plan
        total_nutrients = {
            'Calories': 0,
            'Protein': 0,
            'Carbs': 0,
            'Fat': 0
        }
        for meal in selected_foods.values():
            for food in meal:
                total_nutrients['Calories'] += food.get('calories', 0)
                total_nutrients['Protein'] += food.get('protein', 0)
                total_nutrients['Carbs'] += food.get('carbs', 0)
                total_nutrients['Fat'] += food.get('fat', 0)

        # Get nutrient recommendations
        current_nutrients = predict_current_nutrients(bmi, diet_type)
        target_nutrients = get_target_nutrients(bmi, diet_type)
        recommended_food = "Maintain a balanced diet with a variety of foods."
        if total_nutrients['Calories'] < target_nutrients['Calories']:
            recommended_food = "Consider adding calorie-dense foods like nuts or whole grains."
        elif total_nutrients['Calories'] > target_nutrients['Calories']:
            recommended_food = "Opt for lower-calorie options like vegetables and lean proteins."

        # Store the report in MongoDB
        report = {
            'user_id': session.get('user_id'),
            'bmi': bmi,
            'bmi_category': bmi_category,
            'diet_type': diet_type,
            'nutrients': total_nutrients,
            'recommended_food': recommended_food,
            'selected_foods': selected_foods,
            'created_at': datetime.datetime.utcnow()
        }
        report_id = reports_collection.insert_one(report).inserted_id

        pdf_path = create_pdf(selected_foods, user_data, bmi, bmi_category, nutrition_plot, cluster_plot)

        return jsonify({
            "nutrition_plot": nutrition_plot,
            "cluster_plot": cluster_plot,
            "pdf_path": pdf_path,
            "report_id": str(report_id)
        })

    except Exception as e:
        print(f"Error in generate-plan: {e}")
        return jsonify({"error": f"An error occurred while generating your meal plan. Please try again. Details: {str(e)}"}), 500

@app.route('/get-reports', methods=['GET'])
def get_reports():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "User not logged in"}), 401

        reports = list(reports_collection.find({'user_id': session['user_id']}).sort('created_at', -1))
        reports_data = []
        for report in reports:
            reports_data.append({
                'id': str(report['_id']),
                'date': report['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                'bmi': report['bmi'],
                'nutrients': report['nutrients']
            })
        return jsonify({"reports": reports_data})

    except Exception as e:
        print(f"Error in get-reports: {e}")
        return jsonify({"error": "Failed to fetch reports", "details": str(e)}), 500

@app.route('/compare-reports', methods=['POST'])
def compare_reports():
    try:
        request_data = request.get_json()
        report_id = request_data.get('report_id')
        current_report_id = request_data.get('current_report_id')

        if not report_id or not current_report_id:
            return jsonify({"error": "Missing report IDs"}), 400

        previous_report = reports_collection.find_one({'_id': ObjectId(report_id)})
        current_report = reports_collection.find_one({'_id': ObjectId(current_report_id)})

        if not previous_report or not current_report:
            return jsonify({"error": "One or both reports not found"}), 404

        # Generate comparison plot
        comparison_plot = generate_comparison_plot(previous_report, current_report, "Previous vs Current Nutrients")

        # Generate inspiring message based on nutrient changes
        inspiring_message = ""
        nutrient_changes = {}
        for nutrient in ['Calories', 'Protein', 'Carbs', 'Fat']:
            previous_value = previous_report['nutrients'].get(nutrient, 0)
            current_value = current_report['nutrients'].get(nutrient, 0)
            change = current_value - previous_value
            nutrient_changes[nutrient] = change

        # Determine the overall progress
        bmi_improved = current_report['bmi'] < previous_report['bmi'] if previous_report['bmi'] >= 25 else current_report['bmi'] > previous_report['bmi']
        if bmi_improved:
            inspiring_message = "Great job! Your BMI is moving in the right direction. Keep up the amazing work!"
        else:
            inspiring_message = "You're making progress! Let's adjust your plan to get closer to your goals."

        # Add nutrient-specific messages
        messages = []
        for nutrient, change in nutrient_changes.items():
            if change > 0:
                messages.append(f"Your {nutrient} intake increased by {change:.1f}. Nice effort!")
            elif change < 0:
                messages.append(f"Your {nutrient} intake decreased by {abs(change):.1f}. Let's find a balance!")
            else:
                messages.append(f"Your {nutrient} intake remained stable. Consistency is key!")

        inspiring_message += " " + " ".join(messages)

        return jsonify({
            "comparison_plot": comparison_plot,
            "inspiring_message": inspiring_message,
            "previous_bmi": previous_report['bmi'],
            "current_bmi": current_report['bmi']
        })

    except Exception as e:
        print(f"Error in compare-reports: {e}")
        return jsonify({"error": "Failed to compare reports", "details": str(e)}), 500

@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    try:
        pdf_path = request.args.get('path')
        if not pdf_path or not os.path.exists(pdf_path):
            return jsonify({"error": "PDF file not found"}), 404

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name='personalized_meal_plan.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        print(f"Error in download-pdf: {e}")
        return jsonify({"error": "Failed to download PDF", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)