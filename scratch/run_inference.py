import pathlib, sys, json
# Adjust path to repository root and add src
_repo_root = pathlib.Path(__file__).resolve().parents[1]  # scratch -> repo root
_src_dir = _repo_root / 'src'
print('Adding to sys.path:', str(_src_dir))
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
print('sys.path[0]', sys.path[0])

import torch
from models.classical.model import DenseNet121MultiLabel
from preprocessing.transforms import DeterministicPreprocessor
from models.classical.xai import gradcam
from preprocessing.labels import TARGET_LABELS

ckpt_path = pathlib.Path('E:/College/QRadAssist/src/models/classical/best_model.pt')
model = DenseNet121MultiLabel(pretrained=False)
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
if isinstance(state, dict) and 'model_state_dict' in state:
    state = state['model_state_dict']
model.load_state_dict(state)
model.eval()

img_path = pathlib.Path('E:/College/QRadAssist/data/raw/chexpert-small/train/patient00001/study1/view1_frontal.jpg')
preprocessor = DeterministicPreprocessor()
arr = preprocessor(img_path)
print('preprocess shape', arr.shape)

tensor = torch.from_numpy(arr).unsqueeze(0).float()
with torch.no_grad():
    logits = model(tensor)
    probs = torch.sigmoid(logits)
print('Raw logits:', logits.squeeze().tolist())
print('Sigmoid probs:', probs.squeeze().tolist())
for idx, disease in enumerate(TARGET_LABELS):
    hm = gradcam(model, tensor, idx)
    print(f'GradCAM {disease}: shape {hm.shape}, min {hm.min().item():.6f}, max {hm.max().item():.6f}')
