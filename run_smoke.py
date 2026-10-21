import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add project root to path
from src.models.classical.training import train

train_manifest = Path('data/splits/train_manifest.csv')
val_manifest = Path('data/splits/val_manifest.csv')

model, val_auc, val_ap = train(
    train_manifest=train_manifest,
    val_manifest=val_manifest,
    batch_size=4,
    epochs=1,
    max_train_samples=200,
)
print('Smoke training completed')
print('Validation AUROC sample:', {k: v for k, v in val_auc.items() if v is not None})
