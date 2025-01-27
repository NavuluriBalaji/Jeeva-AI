from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define pipelines for each disease
pipelines = {
    "brain_tumor": pipeline("image-classification", model="Devarshi/Brain_Tumor_Classification"),
    "breast_cancer": pipeline("image-classification", model="amanvvip2/finetuned-breast_cancer_images"),
    "skin_cancer": pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files or "model" not in request.form:
        return "No file or model selected", 400

    file = request.files["file"]
    model = request.form["model"]

    if file.filename == "":
        return "No selected file", 400

    if file and model in pipelines:
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            return "Invalid file format. Only .png, .jpg, .jpeg are allowed.", 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load the image and classify
        image = Image.open(filepath)
        prediction = pipelines[model](image)

        # Cleanup uploaded file
        os.remove(filepath)

        # Return result
        return render_template("result.html", prediction=prediction, model=model)

    return "Invalid request", 400

@app.route("/result")
def result():
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
