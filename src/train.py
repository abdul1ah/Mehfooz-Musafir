import os
import shutil
import sys
from ultralytics import YOLO

# pathing set-up

# 1. Drive Paths 
DRIVE_ROOT = '/content/drive/MyDrive/Mehfooz Musafir'
ZIP_PATH = os.path.join(DRIVE_ROOT, 'data', 'dataset.zip')
SAVE_DIR = os.path.join(DRIVE_ROOT, 'logs')  # YOLO will save runs here
FINAL_WEIGHTS_DIR = os.path.join(DRIVE_ROOT, 'weights')

# 2. Local Paths (Where the GPU will crunch data - Fast SSD/RAM)
# We use /content/dataset because it matches your data.yaml
LOCAL_DATASET_DIR = '/content/dataset' 


# setting up environment

def setup_environment():
    """
    Unzips data and fixes the 'Folder-in-a-Folder' (Matryoshka) problem.
    Ensures data.yaml is always at /content/dataset/data.yaml
    """
    print(f"[Setup] Checking data in {LOCAL_DATASET_DIR}...")

    # 1. Check if data is already ready (Fast Path)
    if os.path.exists(os.path.join(LOCAL_DATASET_DIR, 'data.yaml')):
        print("[Setup] Data ready and flattened. Skipping unzip.")
        return

    # 2. Unzip if needed
    if os.path.exists(ZIP_PATH):
        if not os.path.exists(LOCAL_DATASET_DIR):
            os.makedirs(LOCAL_DATASET_DIR)
            
        print(f"[Setup] Unzipping dataset...")
        # Unzip quiet (-q) to the local directory
        exit_code = os.system(f'unzip -q "{ZIP_PATH}" -d "{LOCAL_DATASET_DIR}"')
        
        if exit_code != 0:
            print("[Setup] Unzip failed!")
            sys.exit(1)
    else:
        print(f"[Setup] Zip file not found at: {ZIP_PATH}")
        sys.exit(1)

    # 3. THE FIX: Flatten the directory structure
    # Check if the unzip created a nested 'dataset' folder
    nested_dir = os.path.join(LOCAL_DATASET_DIR, 'dataset')
    
    if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
        print("[Setup] Detected nested folder structure. Flattening...")
        
        # Move everything from /content/dataset/dataset/* to /content/dataset/*
        for filename in os.listdir(nested_dir):
            source = os.path.join(nested_dir, filename)
            destination = os.path.join(LOCAL_DATASET_DIR, filename)
            
            # Move it (Overwrite if exists)
            shutil.move(source, destination)
            
        # Delete the empty nested folder
        os.rmdir(nested_dir)
        print("[Setup] Flattening complete. Data structure is fixed.")
    else:
        print("[Setup] Directory structure looks correct.")


# training

def train():
    print("[Train] Initializing YOLOv8 Small...")
    
    # Load model
    model = YOLO('yolov8s.pt') 

    # Start Training
    # We save directly to Drive (project=SAVE_DIR) so checkpoints are safe instantly.
    results = model.train(
        data=f'{LOCAL_DATASET_DIR}/data.yaml',
        project=SAVE_DIR,                # Saves to /content/drive/.../logs
        name='mehfooz_run3',             # Creates /logs/mehfooz_run/
        epochs=100,                      
        imgsz=640,
        batch=16,                        # Safe bet for T4 GPU
        patience=10,                     # Stop if no improvement
        exist_ok=True,                   # Overwrite previous run if name is same (or change name)
        
        # PERFORMANCE & SAVING ARGS
        amp=True,                        # Mixed Precision (fp16) - 2x Speed!
        save=True,                       # Saves last.pt and best.pt every epoch
        workers=8,                       # Number of CPU threads for data loading
        cache=False,                     # Set True if you have massive RAM, False for Colab Free Tier

        # DATA AUGMENTATION ARGS (Added)
        degrees=15.0,                    # Rotate images ±15 degrees
        fliplr=0.5,                      # Flip images left-right 50% of the time
        mixup=0.15,                      # Blend images together (helps small datasets)
        copy_paste=0.1,                  # Randomly copy objects to new locations
        mosaic=1.0                       # Mosaic augmentation (default is 1.0, keeping it explicit)
    )

    print("[Train] Training Finished.")
    
    # post-training: copy best.pt to a clean weights folder

    # Copy the best model to your clean 'weights' folder for easy download
    # Note: Ensure this path matches the 'name' used above if you want the new file
    best_weight_path = os.path.join(SAVE_DIR, 'mehfooz_run', 'weights', 'best.pt')
    target_path = os.path.join(FINAL_WEIGHTS_DIR, 'best.pt')

    if os.path.exists(best_weight_path):
        print(f"[Save] Copying best.pt to {FINAL_WEIGHTS_DIR}...")
        os.makedirs(FINAL_WEIGHTS_DIR, exist_ok=True)
        shutil.copy(best_weight_path, target_path)
        print("[Save] Done. You can download it now.")
    else:
        print("[Warning] Could not find best.pt to copy.")
        
if __name__ == '__main__':
    setup_environment()
    train()