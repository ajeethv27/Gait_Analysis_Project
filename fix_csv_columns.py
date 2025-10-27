"""
Fix Oscilloscope CSV Files - Column Headers
============================================
Fixes CSVs with invalid column names to standard format:
Time(s), Voltage(V)

Run this in your User_Data_Labelled/ folder
"""

import pandas as pd
import glob
import os
from pathlib import Path

def fix_csv_columns(folder_path='User_Data_Labelled'):
    """
    Fix column names in oscilloscope CSV files
    
    Handles various possible column name formats:
    - time, voltage → Time(s), Voltage(V)
    - Time, Voltage → Time(s), Voltage(V)
    - col1, col2 → Time(s), Voltage(V)
    """
    
    print("="*70)
    print("FIXING OSCILLOSCOPE CSV COLUMN HEADERS")
    print("="*70)
    
    csv_files = glob.glob(str(Path(folder_path) / '*.csv'))
    
    if not csv_files:
        print(f"⚠️  No CSV files found in {folder_path}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    fixed_count = 0
    error_count = 0
    already_ok = 0
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Check if already correct
            if list(df.columns) == ['Time(s)', 'Voltage(V)']:
                already_ok += 1
                continue
            
            # Try to fix columns
            if len(df.columns) >= 2:
                # Take first two columns, rename them
                df_fixed = df.iloc[:, :2].copy()
                df_fixed.columns = ['Time(s)', 'Voltage(V)']
                
                # Backup original
                backup_path = file_path + '.backup'
                os.rename(file_path, backup_path)
                
                # Save fixed version
                df_fixed.to_csv(file_path, index=False)
                
                print(f"  ✓ Fixed: {filename}")
                fixed_count += 1
            else:
                print(f"  ✗ Error: {filename} - Less than 2 columns")
                error_count += 1
                
        except Exception as e:
            print(f"  ✗ Error: {filename} - {str(e)}")
            error_count += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Already correct: {already_ok}")
    print(f"Fixed: {fixed_count}")
    print(f"Errors: {error_count}")
    
    if fixed_count > 0:
        print(f"\n✓ Backups saved as *.csv.backup")
        print(f"✓ Run validate_dataset.py again to verify")
    
    if error_count == 0 and fixed_count + already_ok == len(csv_files):
        print("\n✅ ALL FILES NOW HAVE CORRECT COLUMNS!")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    
    import sys
    
    # Get folder path from command line or use default
    folder = sys.argv[1] if len(sys.argv) > 1 else 'User_Data_Labelled'
    
    fix_csv_columns(folder)
