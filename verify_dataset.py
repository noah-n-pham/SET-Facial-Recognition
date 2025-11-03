"""
Verify dataset structure and count images per person.
Expected: 9 people × ~100 images = ~900 total
"""
import os
from pathlib import Path

def verify_dataset():
    dataset_path = Path("data/raw/Dataset")
    
    # TODO: Check if dataset path exists
    # If not, print error message and return
    
    print("="*50)
    print("Dataset Verification")
    print("="*50)
    
    # TODO: Get list of all subdirectories (people names)
    # Use dataset_path.iterdir() and filter for directories
    # Sort the list for consistent output
    
    total_images = 0
    
    # TODO: For each person folder:
    #   1. Get path to person's folder
    #   2. Count PNG and JPG files (use glob("*.png") and glob("*.jpg"))
    #   3. Add count to total_images
    #   4. Print person name and count
    #   5. Show ✅ if count >= 50, otherwise show ⚠️
    
    print("="*50)
    # TODO: Print total number of people and total images
    
    # TODO: Check if dataset is complete:
    #   - Should have 9 people
    #   - Should have >= 800 total images
    #   - Print ✅ if good, ⚠️ if incomplete

if __name__ == "__main__":
    verify_dataset()

