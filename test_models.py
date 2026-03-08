import pickle
from tensorflow.keras.models import load_model

def test_models():
    print("Testing pickle models...")
    for model_name in ['diabetes.pkl', 'breast_cancer.pkl', 'heart.pkl', 'kidney.pkl', 'liver.pkl']:
        try:
            with open(f'models/{model_name}', 'rb') as f:
                model = pickle.load(f)
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    print("\nTesting keras models...")
    for model_name in ['malaria.keras', 'pneumonia.keras']:
        try:
            model = load_model(f'models/{model_name}')
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

if __name__ == '__main__':
    test_models()
