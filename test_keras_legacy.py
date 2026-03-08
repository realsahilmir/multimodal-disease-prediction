import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
try:
    from tf_keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

def test_models():
    print("\nTesting keras models with tf_keras...")
    for model_name in ['malaria.h5', 'pneumonia.h5']:
        try:
            model = load_model(f'models/{model_name}')
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

if __name__ == '__main__':
    test_models()
