import cv2
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display required)
import matplotlib.pyplot as plt

# MediaPipe Face Mesh landmark indices for cheek regions
LEFT_CHEEK_INDICES = [
    205, 50, 117, 118, 101, 36, 205,
    123, 147, 213, 192, 214, 135, 138,
    177, 215, 137, 227, 34, 93, 132,
    58, 172, 136, 150, 149, 176, 148
]

RIGHT_CHEEK_INDICES = [
    425, 280, 346, 347, 330, 266, 425,
    352, 376, 433, 416, 434, 364, 367,
    401, 435, 366, 447, 264, 323, 361,
    288, 397, 365, 379, 378, 400, 377
]

def load_landmarks(json_path):
    """Load landmarks from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_cheek_landmarks(landmarks_data, cheek_indices):
    """Extract specific landmark points for a cheek region"""
    cheek_points = []
    for idx in cheek_indices:
        if idx < len(landmarks_data):
            lm = landmarks_data[idx]
            cheek_points.append([lm['x'], lm['y']])
    return np.array(cheek_points, dtype=np.int32)

def create_cheek_mask(image_shape, cheek_points):
    """Create a binary mask for the cheek region"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(cheek_points)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def calculate_rg_ratio_median(image, mask):
    """
    Calculate the median R/G ratio for pixels in the masked region.
    
    Args:
        image: BGR image
        mask: Binary mask for the region
    
    Returns:
        median_rg_ratio: Median of R/G ratios across all pixels in the region
        num_pixels: Number of pixels in the region
    """
    # Get all pixels in the masked region
    cheek_pixels = image[mask > 0]
    
    if len(cheek_pixels) == 0:
        return None, 0
    
    # Extract BGR channels (OpenCV uses BGR format)
    blue = cheek_pixels[:, 0].astype(np.float32)
    green = cheek_pixels[:, 1].astype(np.float32)
    red = cheek_pixels[:, 2].astype(np.float32)
    
    # Calculate R/G ratio for each pixel
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    rg_ratios = red / (green + epsilon)
    
    # Calculate median
    median_rg_ratio = np.median(rg_ratios)
    
    return median_rg_ratio, len(cheek_pixels)

def extract_rppg_signal():
    """Extract rPPG signal from all frames"""
    
    # Directories
    input_images_dir = 'cropped_faces'
    landmark_data_dir = 'landmark_results/data'
    output_dir = 'rppg_signals'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files (sorted by frame number)
    json_files = sorted([f for f in os.listdir(landmark_data_dir) if f.endswith('.json')])
    
    print(f"Processing {len(json_files)} frames for rPPG signal extraction...")
    print(f"Calculating median R/G ratio for each cheek region\n")
    
    # Store time series data
    frame_numbers = []
    left_cheek_signals = []
    right_cheek_signals = []
    combined_signals = []
    
    for idx, json_file in enumerate(json_files):
        # Load landmark data
        json_path = os.path.join(landmark_data_dir, json_file)
        landmark_data = load_landmarks(json_path)
        
        # Get corresponding image
        image_file = landmark_data['image_file']
        image_path = os.path.join(input_images_dir, image_file)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_file}")
            continue
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Get cheek landmarks
        left_cheek_points = get_cheek_landmarks(landmark_data['landmarks'], LEFT_CHEEK_INDICES)
        right_cheek_points = get_cheek_landmarks(landmark_data['landmarks'], RIGHT_CHEEK_INDICES)
        
        # Create masks
        left_cheek_mask = create_cheek_mask(image.shape, left_cheek_points)
        right_cheek_mask = create_cheek_mask(image.shape, right_cheek_points)
        
        # Calculate R/G ratio median for each cheek
        left_rg_median, left_num_pixels = calculate_rg_ratio_median(image, left_cheek_mask)
        right_rg_median, right_num_pixels = calculate_rg_ratio_median(image, right_cheek_mask)
        
        # Store data
        frame_numbers.append(idx)
        left_cheek_signals.append(left_rg_median if left_rg_median is not None else 0)
        right_cheek_signals.append(right_rg_median if right_rg_median is not None else 0)
        
        # Combined signal (average of both cheeks)
        if left_rg_median is not None and right_rg_median is not None:
            combined = (left_rg_median + right_rg_median) / 2
        elif left_rg_median is not None:
            combined = left_rg_median
        elif right_rg_median is not None:
            combined = right_rg_median
        else:
            combined = 0
        
        combined_signals.append(combined)
        
        print(f"Frame {idx}: Left R/G={left_rg_median:.4f} ({left_num_pixels} px), "
              f"Right R/G={right_rg_median:.4f} ({right_num_pixels} px), "
              f"Combined={combined:.4f}")
    
    # Convert to numpy arrays
    frame_numbers = np.array(frame_numbers)
    left_cheek_signals = np.array(left_cheek_signals)
    right_cheek_signals = np.array(right_cheek_signals)
    combined_signals = np.array(combined_signals)
    
    # Save raw signals
    np.save(os.path.join(output_dir, 'left_cheek_rg_signal.npy'), left_cheek_signals)
    np.save(os.path.join(output_dir, 'right_cheek_rg_signal.npy'), right_cheek_signals)
    np.save(os.path.join(output_dir, 'combined_rg_signal.npy'), combined_signals)
    np.save(os.path.join(output_dir, 'frame_numbers.npy'), frame_numbers)
    
    # Save as JSON for easy viewing
    signal_data = {
        'frame_numbers': frame_numbers.tolist(),
        'left_cheek_rg_median': left_cheek_signals.tolist(),
        'right_cheek_rg_median': right_cheek_signals.tolist(),
        'combined_rg_median': combined_signals.tolist(),
        'total_frames': len(frame_numbers),
        'fps': 30  # As per your detect_faces.py
    }
    
    with open(os.path.join(output_dir, 'rppg_signals.json'), 'w') as f:
        json.dump(signal_data, f, indent=2)
    
    # Plot the signals
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Left cheek
    axes[0].plot(frame_numbers, left_cheek_signals, 'r-', linewidth=1.5)
    axes[0].set_title('Left Cheek R/G Ratio Over Time')
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('R/G Ratio')
    axes[0].grid(True, alpha=0.3)
    
    # Right cheek
    axes[1].plot(frame_numbers, right_cheek_signals, 'g-', linewidth=1.5)
    axes[1].set_title('Right Cheek R/G Ratio Over Time')
    axes[1].set_xlabel('Frame Number')
    axes[1].set_ylabel('R/G Ratio')
    axes[1].grid(True, alpha=0.3)
    
    # Combined signal
    axes[2].plot(frame_numbers, combined_signals, 'b-', linewidth=1.5)
    axes[2].set_title('Combined (Average) R/G Ratio Over Time')
    axes[2].set_xlabel('Frame Number')
    axes[2].set_ylabel('R/G Ratio')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rppg_signals_plot.png'), dpi=150)
    print(f"\nSignal plot saved to: {output_dir}/rppg_signals_plot.png")
    
    # Calculate basic statistics
    print(f"\n{'='*60}")
    print(f"rPPG Signal Extraction Complete!")
    print(f"{'='*60}")
    print(f"Total frames processed: {len(frame_numbers)}")
    print(f"\nLeft Cheek Signal:")
    print(f"  Mean: {np.mean(left_cheek_signals):.4f}")
    print(f"  Std Dev: {np.std(left_cheek_signals):.4f}")
    print(f"  Range: [{np.min(left_cheek_signals):.4f}, {np.max(left_cheek_signals):.4f}]")
    print(f"\nRight Cheek Signal:")
    print(f"  Mean: {np.mean(right_cheek_signals):.4f}")
    print(f"  Std Dev: {np.std(right_cheek_signals):.4f}")
    print(f"  Range: [{np.min(right_cheek_signals):.4f}, {np.max(right_cheek_signals):.4f}]")
    print(f"\nCombined Signal:")
    print(f"  Mean: {np.mean(combined_signals):.4f}")
    print(f"  Std Dev: {np.std(combined_signals):.4f}")
    print(f"  Range: [{np.min(combined_signals):.4f}, {np.max(combined_signals):.4f}]")
    print(f"\nOutputs saved to '{output_dir}/':")
    print(f"  - left_cheek_rg_signal.npy")
    print(f"  - right_cheek_rg_signal.npy")
    print(f"  - combined_rg_signal.npy")
    print(f"  - rppg_signals.json")
    print(f"  - rppg_signals_plot.png")
    print(f"{'='*60}")
    
    return signal_data

if __name__ == "__main__":
    signal_data = extract_rppg_signal()