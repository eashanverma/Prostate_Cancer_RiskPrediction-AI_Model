from flask import Flask, request, redirect, url_for, render_template
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import cv2
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from werkzeug.utils import secure_filename

STATIC_FOLDER = 'static_segapp_color'
os.makedirs(STATIC_FOLDER, exist_ok=True)
app = Flask(__name__, static_folder=STATIC_FOLDER)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
segformer.eval()

# processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
# segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device)
# segformer.eval()

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
model_primary.eval(); model_secondary.eval(); model_fused.eval()

transform_img = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def convert_seg_to_rgb_fixed(seg_mask):
    # Define fixed colors for known class IDs
    FIXED_CLASS_COLORS = {
        0: (0, 0, 0),         # Background
        2: (255, 0, 0),       # Tumor (Red)
        15: (0, 255, 0),      # Benign (Green)
        42: (0, 0, 255),      # Stroma/Other (Blue)
        16: (255, 255, 0),    # Uncertain/Noise (Yellow)
        34: (255, 0, 255)     # Misc Tissue (Magenta)
    }
    
    rgb = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    for cls_id, color in FIXED_CLASS_COLORS.items():
        rgb[seg_mask == cls_id] = color
    return Image.fromarray(rgb)

def overlay_mask_cv2(image_pil, mask_rgb_pil, alpha=0.4):
    image = np.array(image_pil.convert("RGB"))
    mask_rgb = np.array(mask_rgb_pil.convert("RGB").resize(image_pil.size, Image.NEAREST))
    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    return Image.fromarray(overlay)

@app.route('/')
def home():
    return render_template("index.html", results=None)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file or file.filename == '':
        return "No file uploaded."

    original_name = secure_filename(file.filename)
    base_name = os.path.splitext(original_name)[0]
    image_filename = f"{base_name}.png"
    image_path = os.path.join(STATIC_FOLDER, image_filename)

    image_pil = Image.open(file).convert("RGB").resize((512, 512))
    image_pil.save(image_path)
    image_tensor = transform_img(image_pil).to(device)

    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = segformer(**inputs).logits
        seg_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
        seg_mask_resized = Image.fromarray(seg_mask.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
        seg_tensor = transforms.ToTensor()(seg_mask_resized).to(device)

    seg_mask_rgb = convert_seg_to_rgb_fixed(seg_mask)
    seg_rgb_path = os.path.join(STATIC_FOLDER, f"seg_rgb_{base_name}.png")
    seg_mask_rgb.save(seg_rgb_path)

    overlay_img = overlay_mask_cv2(image_pil, seg_mask_rgb, alpha=0.4)
    overlay_path = os.path.join(STATIC_FOLDER, f"overlay_{base_name}.png")
    overlay_img.save(overlay_path)

    input_tensor = torch.cat([image_tensor.unsqueeze(0), seg_tensor.unsqueeze(0)], dim=1)
    with torch.no_grad():
        pred_primary = model_primary(input_tensor)
        pred_secondary = model_secondary(input_tensor)

    primary_grade = torch.argmax(pred_primary, dim=1).item() + 3
    secondary_grade = torch.argmax(pred_secondary, dim=1).item() + 3
    gleason_score = primary_grade + secondary_grade
    risk_class = 0 if gleason_score in [0, 6] else (1 if gleason_score == 7 else 2)

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

    label_map = {
        0: "Low Risk (0 or 6)",
        1: "Intermediate Risk (7)",
        2: "High Risk (8–10)"
    }

    results = f'''
    <h2>Prediction Results</h2>
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

    <h3>Image Results</h3>
    <div class="images">
        <div>
            <h4>Original MRI:</h4>
            <img src="/{STATIC_FOLDER}/{image_filename}" width="256">
        </div>
        <div>
            <h4>SegFormer Mask:</h4>
            <img src="/{STATIC_FOLDER}/seg_rgb_{base_name}.png" width="256">
        </div>
        <div>
            <h4>Overlay:</h4>
            <img src="/{STATIC_FOLDER}/overlay_{base_name}.png" width="256">
        </div>
    </div>

    <h3>Color Legend (SegFormer Output):</h3>
    <ul class="legend">
        <li><span style="color:red;">■</span> Tumor Region</li>
        <li><span style="color:green;">■</span> Benign Tissue</li>
        <li><span style="color:blue;">■</span> Stroma/Other Tissue</li>
        <li><span style="color:yellow;">■</span> Noise/Uncertain Area</li>
        <li><span style="color:magenta;">■</span> Misc Tissue</li>
    </ul>
    '''

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True, port=4000)