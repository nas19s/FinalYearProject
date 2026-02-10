import os
import glob
from bs4 import BeautifulSoup
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
RAW_DIR = os.path.join(PROJECT_ROOT, "01_Data", "raw_sec")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "01_Data", "processed_sec")

# Make sure the processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_and_shrink():
    """
    Reads all raw SEC text files, strips HTML, saves cleaned text, 
    and deletes the original raw files to save space.
    """
    print(f"Scanning {RAW_DIR} for SEC text files...")
    files = glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True)
    print(f"Found {len(files)} files. Starting cleaning process...")

    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Read the raw file
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            # Extract metadata from path
            parts = file_path.split(os.sep)
            ticker = parts[-4]
            report_type = parts[-3]
            filing_id = parts[-2]

            # Strip HTML using BeautifulSoup
            soup = BeautifulSoup(raw_content, "lxml")
            clean_text = soup.get_text(separator=" ", strip=True)

            # Save cleaned text
            new_filename = f"{ticker}_{report_type}_{filing_id}.txt"
            save_path = os.path.join(PROCESSED_DIR, new_filename)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(clean_text)

            # Delete the original raw file to free space
            os.remove(file_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Remove empty folders left behind
    print("Cleaning up empty directories...")
    for root, dirs, _ in os.walk(RAW_DIR, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass  # Folder not empty, skip

    print(f"Processing complete. Cleaned files are in: {PROCESSED_DIR}")

if __name__ == "__main__":
    confirm = input("This will delete the raw SEC files. Proceed? (yes/no): ")
    if confirm.lower() == "yes":
        clean_and_shrink()
    else:
        print("Operation cancelled.")
