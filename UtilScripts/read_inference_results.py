#!/usr/bin/env python3
"""
Script to read and explore inference results from .npz files

Example:
python read_inference_results.py "../AffectGPT/output/results-mer2025ov/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250110100/checkpoint_000030_loss_0.602.npz"
"""

import numpy as np
import os
import sys


def read_inference_results(npz_file_path):
    """
    Read and display inference results from .npz file
    """

    # Check if file exists
    if not os.path.exists(npz_file_path):
        print(f"❌ Error: File '{npz_file_path}' not found!")
        return False

    try:
        # Load the .npz file
        print(f"📂 Loading: {npz_file_path}")
        print("=" * 80)

        data = np.load(npz_file_path, allow_pickle=True)

        # Show what keys are available
        print(f"🔑 Available keys: {list(data.keys())}")
        print()

        # Extract the main results (usually stored as 'name2reason')
        if "name2reason" in data:
            name2reason = data[
                "name2reason"
            ].item()  # .item() converts numpy array back to dict

            print(f"📊 Total samples: {len(name2reason)}")
            print(f"📝 Sample names (first 10): {list(name2reason.keys())[:10]}")
            print()

            # Show a few example results
            print("🔍 Inference results:")
            print("-" * 60)

            for i, (name, response) in enumerate(list(name2reason.items())):
                print(f"\n[{i+1}] Video: {name}")
                print(f"Response: {response}")
                print("-" * 40)

            return name2reason
        else:
            print("⚠️  Warning: 'name2reason' key not found in the file")
            print("Available data:")
            for key in data.keys():
                print(f"  - {key}: {type(data[key])}")

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


if __name__ == "__main__":
    print("🔬 NPZ Inference Results Reader")
    print("=" * 50)

    if len(sys.argv) > 1:
        # File path provided as command line argument
        npz_file = sys.argv[1]
        read_inference_results(npz_file)
    else:
        raise ValueError("No file path provided")
