# scripts/visualize_synthetic_data.py
# Visualize hair strand data from .npz files

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_npz(path):
    """Load npz file and return the data"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    data = np.load(path, allow_pickle=True)
    return data

def visualize_strands_3d(strands, title="Hair Strands Visualization"):
    """Visualize hair strands in 3D"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each strand
    for i, strand in enumerate(strands):
        if len(strand.shape) == 2 and strand.shape[1] >= 3:
            x = strand[:, 0]
            y = strand[:, 1]
            z = strand[:, 2]
            ax.plot(x, y, z, alpha=0.6, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Set equal aspect ratio for better visualization
    max_range = np.array([
        strands[:, :, 0].max() - strands[:, :, 0].min(),
        strands[:, :, 1].max() - strands[:, :, 1].min(),
        strands[:, :, 2].max() - strands[:, :, 2].min()
    ]).max() / 2.0
    
    mid_x = (strands[:, :, 0].max() + strands[:, :, 0].min()) * 0.5
    mid_y = (strands[:, :, 1].max() + strands[:, :, 1].min()) * 0.5
    mid_z = (strands[:, :, 2].max() + strands[:, :, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Adjust camera view (elevation, azimuth)
    ax.view_init(elev=90, azim=-90)  # Looks straight at the hair vertically
    
    plt.tight_layout()
    plt.show()

def visualize_npz(path, fast=False):
    """Load and visualize npz file"""
    print(f"Loading: {path}")
    data = load_npz(path)
    
    if not fast:
        # Print info about the npz file
        print(f"\nContents of {os.path.basename(path)}:")
        print(f"  Keys: {list(data.keys())}")
        for key in data.keys():
            item = data[key]
            if isinstance(item, np.ndarray):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  {key}: {type(item)}")
    
    # Try to find strand data
    strand_data = (
        data.get("strands")
        or data.get("data")
        or data.get("difflocks_output_strands")
        or next((v for v in data.values() if isinstance(v, np.ndarray)), None)
    )
    
    if strand_data is None:
        print("No strand data found.")
        return
    
    if fast and len(strand_data) > 5000:
        strand_data = strand_data[::10]  # Downsample to speed up plotting
        print(f"[FAST MODE] Downsampled to {len(strand_data)} strands for quicker visualization.")
    
    title = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
    visualize_strands_3d(strand_data, title=title)


def main():
    parser = argparse.ArgumentParser(description="Visualize hair strand data from .npz files")
    parser.add_argument("--path", type=str, required=True, help="Path to .npz file to visualize")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (skip logs and downsample strands)")
    args = parser.parse_args()
    
    visualize_npz(args.path, fast=args.fast)

if __name__ == "__main__":
    main()