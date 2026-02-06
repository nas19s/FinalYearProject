import os

TEXT_DIR = "01_Data/processed_sec" # Update this to your path

def check_file_lengths(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    print(f"{filename}: {len(text)} characters")
            except Exception as e:
                print(f"Could not read {filename}: {e}")

check_file_lengths(TEXT_DIR)