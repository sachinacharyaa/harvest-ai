# >>> UPDATED DISEASE CODE >>>
import io
from flask import Flask, render_template, request, redirect, flash
from markupsafe import Markup
from PIL import Image
import torch
from torchvision import transforms
from utils.model import ResNet9
from utils.disease import disease_dic

# If this file is not already present above:
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.pth"

# Disease classes (unchanged; must match training order)
disease_classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load model once
disease_model = ResNet9(3, len(disease_classes))
state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
disease_model.load_state_dict(state)
disease_model.eval()

# Use same transform as original working code (no CenterCrop)
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

def predict_image_bytes(file_bytes: bytes) -> str:
    """Return class name from raw uploaded image bytes."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_t = image_transform(image)
    img_u = img_t.unsqueeze(0)
    with torch.no_grad():
        logits = disease_model(img_u)
        _, preds = torch.max(logits, dim=1)
    return disease_classes[preds.item()]

# Flask app instance (if not already created earlier in file)
app = Flask(__name__)
app.secret_key = "dev-secret"  # needed if using flash()

@app.route("/")
def home():
    return render_template("index.html", title="Plant Disease Detection")

@app.route("/disease-predict", methods=["GET", "POST"])
def disease_prediction():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in request.")
            return redirect(request.url)

        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        try:
            img_bytes = file.read()
            pred_class = predict_image_bytes(img_bytes)
            info_html = disease_dic.get(pred_class, f"No info found for {pred_class}.")
            info_html = Markup(str(info_html))
            return render_template(
                "disease-result.html",
                pred_label=pred_class,
                prediction=info_html,
                title="Disease Result",
            )
        except Exception as e:
            print("Prediction error:", repr(e))  # shows in terminal
            flash("Could not process image. Try another.")
            return redirect(request.url)

    # GET
    return render_template("disease.html", title="Upload Image")
# <<< UPDATED DISEASE CODE <<<
if __name__ == "__main__":
    app.run(debug=True)
