"""
Reference Embeddings Generator - Phase 2B

Builds reference database by:
1. Loading all photos for each person
2. Extracting embeddings from each photo
3. Averaging embeddings per person
4. Saving to models/reference_embeddings.npy

Time: 30-45 minutes
TODOs: 3
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('.')
from models.face_model import FaceEmbeddingModel

MAX_NUM_IMG = 100

def generate_reference_embeddings(dataset_path='data/raw/Dataset',
                                   output_dir='models',
                                   max_images=30):
    """
    Generate reference embeddings from collected photos.
    
    Creates two files:
    - models/reference_embeddings.npy: [num_people, 512] array
    - models/label_names.txt: List of names
    """
    
    print("="*70)
    print("Generating Reference Embeddings")
    print("="*70)
    print()
    
    # TODO 8: Initialize model and find person folders
    # =================================================
    # Steps:
    # 1. Create FaceEmbeddingModel with device='cpu', store as model
    # 2. Convert dataset_path to Path object
    # 3. Find all subdirectories (person folders):
    #    - Use .iterdir() to get all items
    #    - Filter for directories with .is_dir()
    #    - Sort the list for consistent ordering
    # 4. Check if list is empty - print error and return if so
    # 5. Print summary: number of people and image count per person
    #
    # Hints:
    # - Use sorted() and list comprehension to get folder list
    # - Each folder name = person's name
    # - Folder order determines label IDs (first folder = ID 0, etc.)
    
    # Step 1: Initialize the face embedding model
    model = FaceEmbeddingModel(device='cpu')
    
    # Step 2: Convert dataset_path to a Path object
    dataset_path = Path(dataset_path)
    
    # Step 3: Find all subdirectories (person folders)
    person_dirs = sorted([p for p in dataset_path.iterdir() if p.is_dir()])
    
    # Step 4: Check if the list is empty
    if not person_dirs:
        print(f"❌ No person folders found in {dataset_path}")
        return
    
    # Step 5: Print summary
    print(f"✅ Found {len(person_dirs)} people:")
    for person_dir in person_dirs:
        image_count = len([f for f in person_dir.iterdir() if f.is_file()])
        print(f"  - {person_dir.name}: {image_count} images")
    
    reference_embeddings = []
    label_names = []
    
    # TODO 9: Extract and average embeddings for each person
    # =======================================================
    for person_dir in person_dirs:
        label_names.append(person_dir.name)
        person_embeddings = []
        for i in range(MAX_NUM_IMG):
            filePath = Path(f'{person_dir}_{i}.png')
            if(filePath.is_file()):
                image = cv2.imread(filePath)
                embedding = model.extract_embedding(image)
                if (embedding != None):
                    person_embeddings.append(embedding)
        
        embeddings_average = np.mean(person_embeddings, axis=0)
        reference_embeddings.append(np.linalg.norm(embeddings_average))
        print(f'✅ Finished {person_dir.name}')
        
           
    # For each person folder:
    
    # 1. Get person name from folder.named
    # 2. Add name to label_names list
    # 3. Find all .png and .jpg files in folder
    # 4. Limit to max_images
    # 5. Loop through image files:
    #    - Load image with cv2.imread()
    #    - Extract embedding with model.extract_embedding()
    #    - If embedding is not None, add to person_embeddings list
    # 6. Check if person_embeddings is empty (no faces detected)
    # 7. Average embeddings: use np.mean() with axis=0
    # 8. Re-normalize averaged embedding:
    #    - Divide by L2 norm: np.linalg.norm(avg_embedding)
    # 9. Append normalized embedding to reference_embeddings
    # 10. Print progress for this person
    #
    # Hints:
    # - Use tqdm(person_folders, desc="...") for progress bar
    # - axis=0 averages across multiple embeddings
    # - Must re-normalize after averaging!
    # - Skip person if no valid embeddings extracted
    
    
    # TODO 10: Save reference database
    # =================================

    
    # Steps:
    # 1. Convert reference_embeddings list to numpy array with dtype=np.float32
    embeddings = np.array(reference_embeddings, dtype=np.float32)
    
    np.save(f"{output_dir}/reference_embeddings.npy", embeddings)
    # 3. Save embeddings array:
    #    - Use np.save() to save to 'models/reference_embeddings.npy'
    with open(f"{output_dir}/label_names.txt", "w") as file:
        for each in label_names:
            file.write(each + "\n")
    # 4. Save label names to text file:
    #    - Open 'models/label_names.txt' for writing
    #    - Write each name on a new line
    # 5. Print summary:
    #    - File paths
    #    - Array shape
    #    - List of people
    #
    # print(f"Embeddings saved to: {}")
    # print(f"People: {label_names}")
    # Hints:
    # - np.array() converts list to array
    # - Use 'w' mode for writing text file
    # - Write each name with f.write(f"{name}\n")
    
    # raise NotImplementedError("TODO 10: Save reference database")


# Run generator
if __name__ == '__main__':
    dataset_path = Path('data/raw/Dataset')
    
    if not dataset_path.exists():
        print("❌ Dataset folder not found!")
        print("   Run data/face_capture.py first")
        exit(1)
    
    try:
        generate_reference_embeddings()
    except NotImplementedError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
