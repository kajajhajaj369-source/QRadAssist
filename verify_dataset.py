import csv, os
train_path = r'e:/College/QRadAssist/data/splits/train_manifest.csv'
val_path = r'e:/College/QRadAssist/data/splits/val_manifest.csv'
train_ids = set()
val_ids = set()
missing = []
# read train manifest
with open(train_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        train_ids.add(row['patient_id'])
        img_path = os.path.join('e:/College/QRadAssist', row['image_path'])
        if not os.path.isfile(img_path):
            missing.append(('train', row['sample_id'], img_path))
# read val manifest
with open(val_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        val_ids.add(row['patient_id'])
        img_path = os.path.join('e:/College/QRadAssist', row['image_path'])
        if not os.path.isfile(img_path):
            missing.append(('val', row['sample_id'], img_path))
overlap = train_ids.intersection(val_ids)
print(f'Train patients: {len(train_ids)}')
print(f'Val patients: {len(val_ids)}')
print(f'Overlap count: {len(overlap)}')
print(f'Missing files: {len(missing)}')
if missing:
    for m in missing[:10]:
        print('Missing', m)
