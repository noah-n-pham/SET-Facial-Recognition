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

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('.')
from models.face_model import FaceEmbeddingModel


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
    # For each person folder:
    # 1. Get person name from folder.name
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
    
    raise NotImplementedError("TODO 9: Extract and average embeddings")
    
    # TODO 10: Save reference database
    # =================================
    # Steps:
    # 1. Convert reference_embeddings list to numpy array with dtype=np.float32
    # 2. Create output directory with Path.mkdir()
    # 3. Save embeddings array:
    #    - Use np.save() to save to 'models/reference_embeddings.npy'
    # 4. Save label names to text file:
    #    - Open 'models/label_names.txt' for writing
    #    - Write each name on a new line
    # 5. Print summary:
    #    - File paths
    #    - Array shape
    #    - List of people
    #
    # Hints:
    # - np.array() converts list to array
    # - Use 'w' mode for writing text file
    # - Write each name with f.write(f"{name}\n")
    
    raise NotImplementedError("TODO 10: Save reference database")


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
