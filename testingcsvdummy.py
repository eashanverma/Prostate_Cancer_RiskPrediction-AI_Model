import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_img = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

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

def predict_folder(folder_path, output_csv):
    label_map = {0: "Low Risk (0 or 6)", 1: "Intermediate Risk (7)", 2: "High Risk (8–10)"}
    results = []

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in tqdm(image_files, desc="Predicting"):
        path = os.path.join(folder_path, fname)
        image = Image.open(path).convert('RGB')
        image_tensor = transform_img(image).to(device)

        dummy_mask = torch.zeros((1, 512, 512)).to(device)
        input_tensor = torch.cat([image_tensor.unsqueeze(0), dummy_mask.unsqueeze(0)], dim=1)

        with torch.no_grad():
            pred_primary = model_primary(input_tensor)
            pred_secondary = model_secondary(input_tensor)

        primary_grade = torch.argmax(pred_primary).item() + 3
        secondary_grade = torch.argmax(pred_secondary).item() + 3
        gleason_score = primary_grade + secondary_grade

        fused_input = torch.cat([
            image_tensor,
            torch.full((1, 512, 512), primary_grade / 5.0).to(device),
            torch.full((1, 512, 512), secondary_grade / 5.0).to(device),
            torch.full((1, 512, 512), gleason_score / 10.0).to(device)
        ], dim=0).unsqueeze(0)

        with torch.no_grad():
            logits = model_fused(fused_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            risk_class = np.argmax(probs)

        results.append({
            "image": fname,
            "primary_grade": primary_grade,
            "secondary_grade": secondary_grade,
            "gleason_score": gleason_score,
            "predicted_risk_class": risk_class,
            "predicted_risk_label": label_map[risk_class],
            "low_risk_prob": round(probs[0], 4),
            "intermediate_risk_prob": round(probs[1], 4),
            "high_risk_prob": round(probs[2], 4),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    test_dir = "Test_imgs"
    output_csv = "inference_results_dummy.csv"
    predict_folder(test_dir, output_csv)