#!/usr/bin/env python3
"""
Script to create track_all_candidates.csv containing only the 'name' column
from track2_train_mercaptionplus.csv.
"""

import pandas as pd
import os

def create_track_candidates():
    # Input and output file paths
    input_file = "../dataset/mer2025-dataset/track2_train_mercaptionplus.csv"
    output_file = "track_all_candidates.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure the dataset is in the correct location.")
        return False
    
    try:
        # Read the CSV file
        print(f"Reading data from: {input_file}")
        df = pd.read_csv(input_file)
        
        # Check if 'name' column exists
        if 'name' not in df.columns:
            print(f"Error: 'name' column not found in the CSV file!")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Extract only the 'name' column
        name_df = df[['name']].copy()
        
        # Save to new CSV file
        name_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"✅ Successfully created: {output_file}")
        print(f"📊 Total rows: {len(name_df)}")
        print(f"📋 Columns: {list(name_df.columns)}")
        print(f"🔍 Sample entries:")
        print(name_df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating track_all_candidates.csv from MERCaptionPlus dataset...")
    print("=" * 60)
    
    success = create_track_candidates()
    
    if success:
        print("\n✨ Done! The file 'track_all_candidates.csv' has been created.")
    else:
        print("\n❌ Failed to create the file. Please check the error messages above.") 