import cv2
import mediapipe as mp
import os
import json
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Directories
input_dir = 'cropped_faces'
output_dir = 'landmark_results'
output_images_dir = os.path.join(output_dir, 'images')
output_data_dir = os.path.join(output_dir, 'data')

# Create output directories
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_data_dir, exist_ok=True)

# Get all image files
image_files = sorted([f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"Found {len(image_files)} images in {input_dir}")
print("Processing facial landmarks...\n")

# Store all results
all_results = []

# Initialize Face Mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # Includes iris landmarks
    min_detection_confidence=0.5) as face_mesh:
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_dir, image_file)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read: {image_file}")
            continue
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            # Get the first (and should be only) face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates
            h, w, _ = image.shape
            landmarks_data = []
            
            for landmark in face_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z  # Depth information
                landmarks_data.append({
                    'x': x,
                    'y': y,
                    'z': float(z),
                    'visibility': float(landmark.visibility) if hasattr(landmark, 'visibility') else 1.0
                })
            
            # Draw landmarks on image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            # Optionally draw iris landmarks
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            
            # Save annotated image
            output_image_path = os.path.join(output_images_dir, f"landmarks_{image_file}")
            cv2.imwrite(output_image_path, annotated_image)
            
            # Save landmark data as JSON
            result_data = {
                'image_file': image_file,
                'num_landmarks': len(landmarks_data),
                'landmarks': landmarks_data
            }
            
            output_json_path = os.path.join(output_data_dir, f"{os.path.splitext(image_file)[0]}.json")
            with open(output_json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            all_results.append(result_data)
            
            print(f"[{idx+1}/{len(image_files)}] Processed: {image_file} - Found {len(landmarks_data)} landmarks")
        else:
            print(f"[{idx+1}/{len(image_files)}] No face detected: {image_file}")

# Save all results in a single file
summary_path = os.path.join(output_dir, 'all_landmarks.json')
with open(summary_path, 'w') as f:
    json.dump(all_results, f, indent=2)

# Save summary as numpy array for easy loading
if all_results:
    # Create numpy array of all landmarks (num_images x num_landmarks x 3)
    landmarks_array = np.array([r['landmarks'] for r in all_results])
    np_path = os.path.join(output_dir, 'all_landmarks.npy')
    
    # Extract just x, y, z coordinates
    coords_only = np.array([[[lm['x'], lm['y'], lm['z']] for lm in r['landmarks']] 
                            for r in all_results])
    np.save(np_path, coords_only)
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {len(all_results)}/{len(image_files)}")
    print(f"Landmarks per face: 468")
    print(f"\nOutput saved to:")
    print(f"  - Annotated images: {output_images_dir}/")
    print(f"  - Individual JSON files: {output_data_dir}/")
    print(f"  - Summary JSON: {summary_path}")
    print(f"  - NumPy array: {np_path}")
    print(f"{'='*60}")
else:
    print("\nNo faces were detected in any images!")