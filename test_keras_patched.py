import sys
import traceback
from tensorflow.keras.models import load_model

with open("test_results_clean_patched.txt", "w") as f:
    try:
        model1 = load_model('models/malaria_patched.keras')
        f.write("malaria_patched.keras SUCCESS\n")
    except Exception as e:
        f.write("malaria_patched.keras FAILED\n")
        traceback.print_exc(file=f)

    try:
        model2 = load_model('models/pneumonia_patched.keras')
        f.write("pneumonia_patched.keras SUCCESS\n")
    except Exception as e:
        f.write("pneumonia_patched.keras FAILED\n")
        traceback.print_exc(file=f)
