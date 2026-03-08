import sys
import traceback
from tensorflow.keras.models import load_model

with open("test_results_clean.txt", "w") as f:
    try:
        model1 = load_model('models/malaria.keras')
        f.write("malaria.keras SUCCESS\n")
    except Exception as e:
        f.write("malaria.keras FAILED\n")
        traceback.print_exc(file=f)

    try:
        model2 = load_model('models/pneumonia.keras')
        f.write("pneumonia.keras SUCCESS\n")
    except Exception as e:
        f.write("pneumonia.keras FAILED\n")
        traceback.print_exc(file=f)
