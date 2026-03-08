import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

def generate_heuristic_diabetes_model():
    """
    Diabetes Features: pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age
    """
    print("Generating heuristic Diabetes model...")
    # Generate healthy patients
    healthy_X = np.random.uniform(low=[0, 70, 60, 10, 15, 18.0, 0.1, 20], 
                                  high=[2, 110, 80, 25, 100, 24.9, 0.4, 40], size=(500, 8))
    healthy_y = np.zeros(500)
    
    # Generate diabetic patients
    diabetic_X = np.random.uniform(low=[1, 140, 85, 20, 150, 28.0, 0.5, 45], 
                                   high=[8, 200, 110, 45, 300, 45.0, 1.5, 75], size=(500, 8))
    diabetic_y = np.ones(500)
    
    X = np.vstack((healthy_X, diabetic_X))
    y = np.concatenate((healthy_y, diabetic_y))
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    with open('models/diabetes.pkl', 'wb') as f:
        pickle.dump(model, f)
        
def generate_heuristic_cancer_model():
    """Breast Cancer Features (26 generic numerical attributes expected)."""
    print("Generating heuristic Breast Cancer model...")
    # Healthy (lower mean radius/texture/perimeter generally)
    healthy_X = np.random.uniform(low=0.1, high=1.0, size=(500, 26))
    healthy_y = np.zeros(500)
    
    # Malignant (higher mean radius/texture/perimeter generally)
    cancer_X = np.random.uniform(low=0.8, high=2.5, size=(500, 26))
    cancer_y = np.ones(500)
    
    X = np.vstack((healthy_X, cancer_X))
    y = np.concatenate((healthy_y, cancer_y))
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    with open('models/breast_cancer.pkl', 'wb') as f:
        pickle.dump(model, f)
        
def generate_heuristic_heart_model():
    """Heart Disease 13 Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal"""
    print("Generating heuristic Heart Disease model...")
    healthy_X = np.random.uniform(low=[25, 0, 0, 90,  150, 0, 0, 140, 0, 0.0, 0, 0, 1], 
                                  high=[50, 1, 1, 120, 200, 0, 1, 200, 0, 1.0, 1, 0, 2], size=(500, 13))
    healthy_y = np.zeros(500)
    
    heart_X = np.random.uniform(low=[55, 0, 2, 130, 220, 0, 1, 90,  1, 2.0, 1, 1, 2], 
                                high=[85, 1, 3, 180, 350, 1, 2, 130, 1, 4.0, 2, 3, 3], size=(500, 13))
    heart_y = np.ones(500)
    
    X = np.vstack((healthy_X, heart_X))
    y = np.concatenate((healthy_y, heart_y))
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    with open('models/heart.pkl', 'wb') as f:
        pickle.dump(model, f)

def generate_heuristic_kidney_model():
    """Kidney 18 Features"""
    print("Generating heuristic Kidney model...")
    healthy_X = np.random.uniform(low=0.1, high=1.0, size=(500, 18))
    healthy_y = np.zeros(500)
    kidney_X = np.random.uniform(low=0.8, high=2.5, size=(500, 18))
    kidney_y = np.ones(500)
    
    X = np.vstack((healthy_X, kidney_X))
    y = np.concatenate((healthy_y, kidney_y))
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    with open('models/kidney.pkl', 'wb') as f:
        pickle.dump(model, f)

def generate_heuristic_liver_model():
    """Liver 10 Features"""
    print("Generating heuristic Liver model...")
    healthy_X = np.random.uniform(low=0.1, high=1.0, size=(500, 10))
    healthy_y = np.zeros(500)
    liver_X = np.random.uniform(low=0.8, high=2.5, size=(500, 10))
    liver_y = np.ones(500)
    
    X = np.vstack((healthy_X, liver_X))
    y = np.concatenate((healthy_y, liver_y))
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    with open('models/liver.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    generate_heuristic_diabetes_model()
    generate_heuristic_cancer_model()
    generate_heuristic_heart_model()
    generate_heuristic_kidney_model()
    generate_heuristic_liver_model()
