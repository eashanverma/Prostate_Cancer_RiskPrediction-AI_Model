from flask import Flask, request, redirect, url_for, send_from_directory
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static_dummy'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/static_dummy/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_img = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === CNN Models ===
class CNNGradePredictor(nn.Module):
    def __init__(self):
        super(CNNGradePredictor, self).__init__()
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
        super(CNN4RiskClassifier, self).__init__()
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

@app.route('/dummy')
def dummy_home():
    return """
    <h2>Upload Image for Dummy Mask-based Prediction</h2>
    <form method="POST" action="/predict_dummy" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="Predict">
    </form>
    """

# === Prediction with Dummy Mask ===
@app.route('/predict_dummy', methods=['POST'])
def predict_dummy():
    if 'image' not in request.files:
        return redirect(url_for('dummy_home'))

    file = request.files['image']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    image = Image.open(filepath).convert('RGB')
    image_tensor = transform_img(image).to(device)

    # Dummy mask (all zeros)
    dummy_mask = torch.zeros((1, 512, 512)).to(device)
    input_tensor = torch.cat([image_tensor.unsqueeze(0), dummy_mask.unsqueeze(0)], dim=1)

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
        risk_logits = model_fused(fused_input)
        probs = torch.softmax(risk_logits, dim=1).cpu().numpy().flatten()
        final_pred = np.argmax(probs)

    label_map = {0: "Low Risk (0 or 6)", 1: "Intermediate Risk (7)", 2: "High Risk (8–10)"}

    result_html = f"""
    <h2>Prediction Results (Dummy Mask)</h2>
    <p><strong>Primary Grade:</strong> {primary_grade}</p>
    <p><strong>Secondary Grade:</strong> {secondary_grade}</p>
    <p><strong>Gleason Score:</strong> {gleason_score}</p>
    <p><strong>Predicted Risk Class:</strong> {label_map[final_pred]}</p>
    <h3>Confidence Scores:</h3>
    <ul>
        <li>Low Risk: {probs[0]*100:.2f}%</li>
        <li>Intermediate Risk: {probs[1]*100:.2f}%</li>
        <li>High Risk: {probs[2]*100:.2f}%</li>
    </ul>
    <h3>Uploaded MRI:</h3>
    <img src="/static_dummy/{filename}" width="256" style="border:1px solid black;">
    <br><br><a href="/dummy">Try another</a>
    """
    return result_html

if __name__ == "__main__":
    app.run(debug=True, port=5000)