import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize InsightFace with buffalo_l model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Path to your dataset
dataset_path = 'data/raw'
reference_embeddings = {}

# Loop through each person folder
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)

    if embeddings:
        reference_embeddings[person] = np.mean(embeddings, axis=0)

# Save reference embeddings to file
with open('reference_embeddings.pkl', 'wb') as f:
    pickle.dump(reference_embeddings, f)

print("✅ Reference embeddings saved to reference_embeddings.pkl")