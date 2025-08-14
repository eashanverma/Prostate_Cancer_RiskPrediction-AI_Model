import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

TEST_FOLDER = "Test_imgs"
OUTPUT_CSV = "inference_results_segformer.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_img = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_mask = transforms.Compose([
    transforms.Resize((512, 512), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
segformer.eval()

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

results = []

for fname in tqdm(os.listdir(TEST_FOLDER), desc="Running inference"):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(TEST_FOLDER, fname)
    image_pil = Image.open(img_path).convert("RGB")
    image_tensor = transform_img(image_pil).to(device)

    inputs = feature_extractor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segformer(**inputs)
        logits = outputs.logits
        seg_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
        seg_mask = Image.fromarray(seg_mask.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
        seg_tensor = transform_mask(seg_mask).to(device)

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
        risk_logits = model_fused(fused_input)
        probs = torch.softmax(risk_logits, dim=1).cpu().numpy().flatten()
        final_pred = np.argmax(probs)

    results.append({
        "filename": fname,
        "primary_grade": primary_grade,
        "secondary_grade": secondary_grade,
        "gleason_score": gleason_score,
        "predicted_risk_class": final_pred,
        "prob_low": round(probs[0], 4),
        "prob_intermediate": round(probs[1], 4),
        "prob_high": round(probs[2], 4),
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")

import pandas as pd
df = pd.read_csv("inference_results_segformer.csv")
counts = df["predicted_risk_class"].value_counts().sort_index()
for label in [0, 1, 2]:
    print(f"Risk Class {label}: {counts.get(label, 0)}")

# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm
# from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# TEST_FOLDER = "Test_imgs"
# OUTPUT_CSV = "inference_results_segformer_b5.csv"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transform_img = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

# transform_mask = transforms.Compose([
#     transforms.Resize((512, 512), interpolation=Image.NEAREST),
#     transforms.ToTensor()
# ])

# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
# segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device)
# segformer.eval()

# class CNNGradePredictor(nn.Module):
#     def __init__(self):
#         super(CNNGradePredictor, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 3)
#         )

#     def forward(self, x):
#         return self.classifier(self.features(x))

# class CNN4RiskClassifier(nn.Module):
#     def __init__(self):
#         super(CNN4RiskClassifier, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(6, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
#             nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
#             nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3)
#         )

#     def forward(self, x):
#         return self.classifier(self.features(x))

# model_primary = CNNGradePredictor().to(device)
# model_secondary = CNNGradePredictor().to(device)
# model_fused = CNN4RiskClassifier().to(device)

# model_primary.load_state_dict(torch.load("cnn_primary_model4.pth", map_location=device))
# model_secondary.load_state_dict(torch.load("cnn_secondary_model4.pth", map_location=device))
# model_fused.load_state_dict(torch.load("Fused_Final_cnn4.pth", map_location=device))

# model_primary.eval()
# model_secondary.eval()
# model_fused.eval()

# results = []

# for fname in tqdm(os.listdir(TEST_FOLDER), desc="Running inference"):
#     if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
#         continue

#     img_path = os.path.join(TEST_FOLDER, fname)
#     image_pil = Image.open(img_path).convert("RGB")
#     image_tensor = transform_img(image_pil).to(device)

#     inputs = feature_extractor(images=image_pil, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = segformer(**inputs)
#         logits = outputs.logits
#         seg_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
#         seg_mask = Image.fromarray(seg_mask.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
#         seg_tensor = transform_mask(seg_mask).to(device)

#     input_tensor = torch.cat([image_tensor.unsqueeze(0), seg_tensor.unsqueeze(0)], dim=1)

#     with torch.no_grad():
#         pred_primary = model_primary(input_tensor)
#         pred_secondary = model_secondary(input_tensor)

#     primary_grade = torch.argmax(pred_primary, dim=1).item() + 3
#     secondary_grade = torch.argmax(pred_secondary, dim=1).item() + 3
#     gleason_score = primary_grade + secondary_grade

#     if gleason_score in [0, 6]:
#         risk_class = 0
#     elif gleason_score == 7:
#         risk_class = 1
#     else:
#         risk_class = 2

#     fused_input = torch.cat([
#         image_tensor,
#         torch.full((1, 512, 512), primary_grade / 5.0).to(device),
#         torch.full((1, 512, 512), secondary_grade / 5.0).to(device),
#         torch.full((1, 512, 512), gleason_score / 10.0).to(device)
#     ], dim=0).unsqueeze(0)

#     with torch.no_grad():
#         risk_logits = model_fused(fused_input)
#         probs = torch.softmax(risk_logits, dim=1).cpu().numpy().flatten()
#         final_pred = np.argmax(probs)

#     results.append({
#         "filename": fname,
#         "primary_grade": primary_grade,
#         "secondary_grade": secondary_grade,
#         "gleason_score": gleason_score,
#         "predicted_risk_class": final_pred,
#         "prob_low": round(probs[0], 4),
#         "prob_intermediate": round(probs[1], 4),
#         "prob_high": round(probs[2], 4),
#     })

# df = pd.DataFrame(results)
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"Saved predictions to {OUTPUT_CSV}")

# import pandas as pd
# df = pd.read_csv("inference_results_segformer_b5.csv")
# counts = df["predicted_risk_class"].value_counts().sort_index()
# for label in [0, 1, 2]:
#     print(f"Risk Class {label}: {counts.get(label, 0)}")