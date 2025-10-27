"""
Dataset File Validator
======================
This script checks if all 84 expected CSV files for the gait analysis
dataset are present in the target folder.

It reports:
1.  MISSING FILES: Expected files that are not found.
2.  UNEXPECTED FILES: Files found in the folder that are not part of
    the 84 expected filenames (e.g., typos, extra files).

Based on the structure:
-   Subjects: S01 to S12 (12 total)
-   Activities: STAND, WALK, WALK01, RUN, RUN01, JUMP, JUMP01 (7 total)
-   Total: 12 * 7 = 84 files
"""

import sys
from pathlib import Path

# --- Define the "Ground Truth" ---

# 1. List all 12 subjects
SUBJECTS = [f'S{i:02d}' for i in range(1, 13)] # 'S01', 'S02', ..., 'S12'

# 2. List all 7 activities/trials
ACTIVITIES = [
    'STAND', 
    'WALK', 
    'WALK01', 
    'RUN', 
    'RUN01', 
    'JUMP', 
    'JUMP01'
]

# 3. Create the complete set of 84 expected filenames
# We use .upper() to make the check case-insensitive
# This will match 'S01STAND.csv' and 's01stand.CSV'
expected_files = set()
for s in SUBJECTS:
    for a in ACTIVITIES:
        filename = f"{s}{a}.csv" # e.g., "S01STAND.csv"
        expected_files.add(filename.upper()) # Add "S01STAND.CSV"

def validate_dataset_files(folder_path):
    """
    Checks a folder for the complete set of 84 expected CSV files.
    """
    print("="*70)
    print("Validating Dataset Files")
    print(f"Target folder: {folder_path}")
    print("="*70)

    target_dir = Path(folder_path)
    
    # Error handling if the folder doesn't exist
    if not target_dir.is_dir():
        print(f"❌ ERROR: Folder not found at '{folder_path}'")
        print("Please provide the correct path to your data folder.")
        print("\nExample Command:")
        print("  python validate_files.py User_Gait_Data_Master/User_Data_Labelled")
        return

    # --- Get "Actual" Files ---
    # Scan the folder, get all .csv files, and convert names to uppercase
    actual_files = set(f.name.upper() for f in target_dir.glob('*.csv'))
    
    if not actual_files:
        print(f"⚠️ WARNING: No .csv files found in '{folder_path}'.")
        # Continue to show the full list of missing files.

    # --- Compare Sets ---
    missing_files = expected_files - actual_files
    unexpected_files = actual_files - actual_files.intersection(expected_files)
    
    print(f"\nFound {len(actual_files)} CSV files. Expected {len(expected_files)}.")

    # --- Report Results ---
    if not missing_files and not unexpected_files:
        print("\n✅ SUCCESS! All 84 files are present and correctly named.")
        print("="*70)
        return

    # 1. Report Missing Files
    if missing_files:
        print("\n------------------\n"
              f"🔴 MISSING FILES ({len(missing_files)}):"
              "\n------------------")
        # Sort the list for a clean report
        for f in sorted(list(missing_files)):
            print(f"  - {f}")
    
    # 2. Report Unexpected/Badly Named Files
    if unexpected_files:
        print("\n------------------\n"
              f"⚠️  UNEXPECTED / BADLY NAMED FILES ({len(unexpected_files)}):"
              "\n------------------")
        # Sort the list for a clean report
        for f in sorted(list(unexpected_files)):
            print(f"  - {f}")
    
    print("\n" + "="*70)
    print("Check 'UNEXPECTED' list for typos (e.g., 'S01WALK.CSV.backup').")
    print("Check 'MISSING' list for files you need to find or rename.")


if __name__ == "__main__":
    """
    Main entry point for the script.
    """
    # Get folder path from command line argument (e.g., sys.argv[1])
    # If no argument is given, default to the path from your README.
    default_path = 'User_Gait_Data_Master/User_Data_Labelled'
    
    # Use the command-line argument if provided, otherwise use the default
    folder_to_check = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    validate_dataset_files(folder_to_check)
