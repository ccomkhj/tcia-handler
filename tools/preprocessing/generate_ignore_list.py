import pandas as pd
from pathlib import Path
import re
import os

def normalize_patient_id(patient_id):
    if pd.isna(patient_id):
        return None
    if isinstance(patient_id, str):
        if "Biopsy" in patient_id:
            try:
                return int(patient_id.split("-")[-1])
            except ValueError:
                return None
        try:
            return int(patient_id)
        except ValueError:
            return None
    return int(patient_id)

def main():
    input_file = "data/feedback/PIRADS012_gl6.xlsx"
    output_file = "data/ignore_list.parquet"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    xl = pd.ExcelFile(input_file)
    ignore_list = []
    
    possible_cols = [
        "Overlay checked", "Overlay_checked", "Overlay_check", 
        "Overley_check", "Overly_check"
    ]
    
    print(f"Reading {input_file}...")
    
    for sheet in xl.sheet_names:
        if not sheet.startswith("PIRADS"):
            continue
            
        print(f"Processing {sheet}...")
        df = pd.read_excel(xl, sheet_name=sheet)
        
        # Find the check column
        check_col = None
        for col in df.columns:
            if col in possible_cols:
                check_col = col
                break
        
        if not check_col:
            # Try fuzzy match
            for col in df.columns:
                if "overlay" in col.lower() and "check" in col.lower():
                    check_col = col
                    break
        
        if check_col:
            print(f"  Found check column: {check_col}")
            # Filter rows containing "wrong"
            # Handle NaN values
            mask = df[check_col].astype(str).str.contains("wrong", case=False, na=False)
            ignored_df = df[mask]
            
            count = len(ignored_df)
            print(f"  Found {count} patients to ignore in {sheet}")
            
            for _, row in ignored_df.iterrows():
                pid = normalize_patient_id(row['patient_number'])
                if pid is not None:
                    ignore_list.append({
                        'patient_number': pid,
                        'reason': row[check_col],
                        'source_sheet': sheet
                    })
        else:
            print(f"  No check column found in {sheet}")

    if ignore_list:
        ignore_df = pd.DataFrame(ignore_list)
        # Drop duplicates just in case
        ignore_df = ignore_df.drop_duplicates(subset=['patient_number'])
        
        print(f"\nTotal unique patients to ignore: {len(ignore_df)}")
        # print(ignore_df)
        
        ignore_df.to_parquet(output_file, index=False)
        print(f"\nSaved ignore list to {output_file}")
    else:
        print("\nNo patients to ignore found.")

if __name__ == "__main__":
    main()
