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
            # Extract x, y, z coordinates
            x = strand[:, 0]
            y = strand[:, 1]
            z = strand[:, 2]
            
            # Plot the strand as a line
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
    
    plt.tight_layout()
    plt.show()

def visualize_npz(path):
    """Load and visualize npz file"""
    print(f"Loading: {path}")
    data = load_npz(path)
    
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
    # Common keys might be 'strands', 'data', 'points', etc.
    strand_data = None
    
    if 'strands' in data:
        strand_data = data['strands']
    elif 'data' in data:
        strand_data = data['data']
    elif len(data.keys()) == 1:
        # If there's only one key, use that
        strand_data = data[list(data.keys())[0]]
    else:
        # Try the first array-like key
        for key in data.keys():
            if isinstance(data[key], np.ndarray) and len(data[key].shape) >= 2:
                strand_data = data[key]
                print(f"\nUsing key '{key}' for visualization")
                break
    
    if strand_data is None:
        print("\nCould not find suitable strand data to visualize")
        print("Available keys:", list(data.keys()))
        return
    
    # Get the filename for the title
    filename = os.path.basename(path)
    folder = os.path.basename(os.path.dirname(path))
    title = f"{folder}/{filename}\nStrands: {len(strand_data)}"
    
    print(f"\nVisualizing {len(strand_data)} strands...")
    print("Close the window to exit.")
    
    visualize_strands_3d(strand_data, title=title)

def main():
    parser = argparse.ArgumentParser(description="Visualize hair strand data from .npz files")
    parser.add_argument("--path", type=str, required=True, help="Path to .npz file to visualize")
    args = parser.parse_args()
    
    if not args.path.endswith('.npz'):
        print("Warning: File does not have .npz extension")
    
    visualize_npz(args.path)

if __name__ == "__main__":
    main()