from flask import Flask, request, redirect, url_for
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from werkzeug.utils import secure_filename

STATIC_FOLDER = 'static_segapp_normal'
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
segformer.eval()

class CNNGradePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class CNN4RiskClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model_primary = CNNGradePredictor().to(device)
model_secondary = CNNGradePredictor().to(device)
model_fused = CNN4RiskClassifier().to(device)

model_primary.load_state_dict(torch.load("cnn_primary_model4.pth", map_location=device))
model_secondary.load_state_dict(torch.load("cnn_secondary_model4.pth", map_location=device))
model_fused.load_state_dict(torch.load("Fused_Final_cnn4.pth", map_location=device))

model_primary.eval()
model_secondary.eval()
model_fused.eval()

transform_img = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route('/')
def home():
    return """
    <h2>Upload MRI for Risk Prediction</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="Predict">
    </form>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']
    original_name = secure_filename(file.filename)
    base_name = os.path.splitext(original_name)[0]

    image_filename = f"{base_name}.png"
    image_path = os.path.join(STATIC_FOLDER, image_filename)

    image_pil = Image.open(file).convert('RGB').resize((512, 512))
    image_pil.save(image_path)
    image_tensor = transform_img(image_pil).to(device)

    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segformer(**inputs)
        seg_mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
        seg_mask_pil = Image.fromarray(seg_mask.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
        seg_tensor = transforms.ToTensor()(seg_mask_pil).to(device)

    mask_filename = f"mask_{base_name}.png"
    mask_path = os.path.join(STATIC_FOLDER, mask_filename)
    seg_mask_pil.save(mask_path)

    input_tensor = torch.cat([image_tensor.unsqueeze(0), seg_tensor.unsqueeze(0)], dim=1)
    with torch.no_grad():
        pred_primary = model_primary(input_tensor)
        pred_secondary = model_secondary(input_tensor)

    primary_grade = torch.argmax(pred_primary, dim=1).item() + 3
    secondary_grade = torch.argmax(pred_secondary, dim=1).item() + 3
    gleason_score = primary_grade + secondary_grade

    if gleason_score in [0, 6]:
        risk_class = 0
    elif gleason_score == 7:
        risk_class = 1
    else:
        risk_class = 2

    fused_input = torch.cat([
        image_tensor,
        torch.full((1, 512, 512), primary_grade / 5.0).to(device),
        torch.full((1, 512, 512), secondary_grade / 5.0).to(device),
        torch.full((1, 512, 512), gleason_score / 10.0).to(device)
    ], dim=0).unsqueeze(0)

    with torch.no_grad():
        fused_out = model_fused(fused_input)
        probs = torch.softmax(fused_out, dim=1).cpu().numpy().flatten()
        final_pred = np.argmax(probs)

    label_map = {0: "Low Risk (0 or 6)", 1: "Intermediate Risk (7)", 2: "High Risk (8–10)"}

    return f"""
    <h2>Prediction Results (with SegFormer)</h2>
    <p><strong>Primary Grade:</strong> {primary_grade}</p>
    <p><strong>Secondary Grade:</strong> {secondary_grade}</p>
    <p><strong>Gleason Score:</strong> {gleason_score}</p>
    <p><strong>Predicted Risk Class:</strong> {label_map[final_pred]}</p>
    <h3>Class Probabilities:</h3>
    <ul>
        <li>Low Risk: {probs[0]*100:.2f}%</li>
        <li>Intermediate Risk: {probs[1]*100:.2f}%</li>
        <li>High Risk: {probs[2]*100:.2f}%</li>
    </ul>
    <h3>Uploaded MRI:</h3>
    <img src="/static_segapp_normal/{image_filename}" width="256" style="border:2px solid black;"><br><br>
    <h3>SegFormer Generated Mask:</h3>
    <img src="/static_segapp_normal/{mask_filename}" width="256" style="border:2px solid red;"><br><br>
    <a href="/">Try another image</a>
    """

if __name__ == "__main__":
    app.run(debug=True, port=8000)