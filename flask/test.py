from ml import predict, dump_model
import os

model_path = "model_save.pkl"
if not os.path.exists(model_path):
    dump_model()
else:
    try:
        predict(["test"])  # Attempt to load the model
    except Exception as e:
        print(f"Error loading model: {e}. Retraining the model.")
        dump_model()

syms = ["mild fever", "chills", "watering from eyes", "headache"]
print(predict(syms))