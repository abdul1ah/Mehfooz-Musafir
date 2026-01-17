import os
import shutil
import sys
from ultralytics import YOLO

# --- PATH CONFIGURATION (Matches train.py) ---
# 1. Drive Paths
DRIVE_ROOT = '/content/drive/MyDrive/Mehfooz Musafir'
ZIP_PATH = os.path.join(DRIVE_ROOT, 'data', 'dataset.zip')
LOGS_DIR = os.path.join(DRIVE_ROOT, 'logs')

# 2. Local Paths (Fast SSD)
LOCAL_DATASET_DIR = '/content/dataset'
DATA_YAML = f'{LOCAL_DATASET_DIR}/data.yaml'

# 3. Model Path (Updated to your renamed file)
WEIGHTS_PATH = os.path.join(DRIVE_ROOT, 'weights', 'best_run4.pt')

# --- ENVIRONMENT SETUP (Copied from train.py) ---
def setup_environment():
    """
    Unzips data and fixes the 'Folder-in-a-Folder' problem.
    Ensures data.yaml is always at /content/dataset/data.yaml
    """
    print(f"🛠️ [Setup] Checking data in {LOCAL_DATASET_DIR}...")

    # 1. Check if data is already ready (Fast Path)
    if os.path.exists(DATA_YAML):
        print("✅ [Setup] Data ready and flattened. Skipping unzip.")
        return

    # 2. Unzip if needed
    if os.path.exists(ZIP_PATH):
        if not os.path.exists(LOCAL_DATASET_DIR):
            os.makedirs(LOCAL_DATASET_DIR)
            
        print(f"📦 [Setup] Unzipping dataset from Drive...")
        # Unzip quiet (-q) to the local directory
        exit_code = os.system(f'unzip -q "{ZIP_PATH}" -d "{LOCAL_DATASET_DIR}"')
        
        if exit_code != 0:
            print("❌ [Setup] Unzip failed!")
            sys.exit(1)
    else:
        print(f"❌ [Setup] Zip file not found at: {ZIP_PATH}")
        sys.exit(1)

    # 3. THE FIX: Flatten the directory structure
    # Check if the unzip created a nested 'dataset' or 'data' folder
    # Sometimes zips contain 'data/train' instead of just 'train'
    nested_dirs = ['dataset', 'data'] # Check for common nested names
    
    for nested_name in nested_dirs:
        nested_path = os.path.join(LOCAL_DATASET_DIR, nested_name)
        if os.path.exists(nested_path) and os.path.isdir(nested_path):
            print(f"⚠️ [Setup] Detected nested folder '{nested_name}'. Flattening...")
            
            # Move everything up one level
            for filename in os.listdir(nested_path):
                source = os.path.join(nested_path, filename)
                destination = os.path.join(LOCAL_DATASET_DIR, filename)
                
                # Move it (Overwrite if exists)
                if os.path.exists(destination):
                    if os.path.isdir(destination):
                        shutil.rmtree(destination)
                    else:
                        os.remove(destination)
                shutil.move(source, destination)
                
            # Delete the empty nested folder
            os.rmdir(nested_path)
            print("✅ [Setup] Flattening complete. Data structure is fixed.")
            break # Stop checking other names if one was found

# --- VALIDATION LOGIC ---
def validate():
    print(f"🔍 [Test] Checking model: {WEIGHTS_PATH}")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ [Error] Model not found at: {WEIGHTS_PATH}")
        return

    # Load the model
    model = YOLO(WEIGHTS_PATH)

    print("👨‍🏫 [Test] Starting Final Evaluation on the TEST set...")

    # Run Validation on 'test' split
    metrics = model.val(
        data=DATA_YAML,
        split='test',        # 👈 Forces use of 'test' folder
        project=LOGS_DIR,
        name='mehfooz_final_test_exam',
        exist_ok=True,
        plots=True
    )

    # Print Report
    print("\n" + "="*40)
    print("🏆 FINAL TEST RESULTS")
    print("="*40)
    print(f"Overall mAP50:    {metrics.box.map50:.3f}")
    print(f"Overall mAP50-95: {metrics.box.map:.3f}")
    print("="*40)
    print(f"📄 Full detailed report saved to: {LOGS_DIR}/mehfooz_final_test_exam")

# --- EXECUTE ---
if __name__ == "__main__":
    setup_environment()
    validate()