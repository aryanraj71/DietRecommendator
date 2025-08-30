# Diet Recommendation System using Machine Learning

This is a machine learning-powered diet recommendation web app that suggests personalized food options based on a user’s BMI (Body Mass Index), nutritional needs, and taste preferences. It uses both supervised and unsupervised learning models to classify and cluster food data and provides downloadable diet reports in PDF format.

---

##  Features

- BMI calculation based on user input (height and weight)
- Intelligent food recommendation using:
  - 🔹 **Random Forest** for calorie-level classification
  - 🔹 **KMeans Clustering** for nutrient-based grouping
- Dynamic food suggestions based on underweight/normal/overweight category
- Option to select preferred food items
- Download personalized diet plan in PDF

---

## Machine Learning Used

### Random Forest Classifier (Supervised Learning)
- Used to classify food into **five calorie levels**
- Labels (Y) generated using `pandas.qcut()` based on calorie values
- Inputs (X): `Calories`, `Protein`, `Carbs`, `Fat`

### KMeans Clustering (Unsupervised Learning)
- Used to group food items with similar nutritional profiles
- Helps recommend **similar alternatives** if required items are unavailable
- Inputs (X): `Calories`, `Protein`, `Carbs`, `Fat`

### PCA (Principal Component Analysis)
- Used to reduce feature dimensions from 4D to 2D for visualization

---

## Dataset: `food.csv`

-  **Total Features:** 16
-  **Labelled Columns (12):** For filtering (e.g., Meal category, Cuisine type, etc.)
-  **Unlabelled Columns (4):** Used for ML: `Calories`, `Protein`, `Carbs`, `Fat`
-  **Target Variable for RF:** `Calorie_Level` (created artificially)

---

##  Tech Stack

| Layer       | Tools Used                    |
|-------------|-------------------------------|
| Backend     | Python, Flask, scikit-learn   |
| Frontend    | HTML, CSS, JS (Vanilla)       |
| Data Viz    | matplotlib, seaborn           |
| Data PDF    | fpdf                           |

---

## Project Flow

1. User opens the web page and enters:
   - Age (not used), Gender, Height, Weight, Meal time
2. BMI is calculated and category is determined
3. Random Forest suggests food with matching calorie levels
4. If suggestions are limited, KMeans finds nutrient-similar food
5. User selects food → Nutritional info shown
6. Final report is downloadable as PDF

---

## Folder Structure

DietRecommendator/
├── app.py                         
├── food.csv                       
├── requirements.txt               
├── static/                        
│   ├── login-bg.jpg             
│   └── logo.png
│   └── nutrition.jpg             
├── templates/                     
│   ├── index.html                 
│   ├── front.html                  
│   ├── login.html 
│   └── signup.htm
└── README.md        


---

## How to Run


1. Install requirements  
   `pip install -r requirements.txt`

2. Run the app  
   `python app.py`

3. Open browser  
   Visit `http://localhost:5000/`

---

##  Author

- Project by: Aryan Raj 
- College/Institution: UPES, Dehradun

- Guided by: Dr. Neeraj Chugh
