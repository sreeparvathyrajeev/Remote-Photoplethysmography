# Remote Photoplethysmography (rPPG) Face Detection Models

| Model/Algorithm | Type of Detection | Advantages | Limitations |
| :--- | :--- | :--- | :--- |
| **Viola and Jones (VJ) / Haar Cascade** | Face Detection (Bounding Box) | **Simple and Fast**. Widely available in libraries like OpenCV. | Inaccurate bounding box (includes non-facial pixels). **Computationally expensive** to run on every frame. Vulnerable to head pose, rotation, and occlusion. |
| **Kanade-Lucas-Tomasi (KLT)** | Feature/Face Tracking | **Reduces computational cost** when combined with VJ. Can track slight movements. | Only re-detection is needed for **severe motion**. |
| **Deep Learning Models** (e.g., BlazeFace, SSD, CNN-based) | Face Detection & Landmark/Key Point Detection | **High Accuracy & Robustness** over conventional methods. Optimized for **real-time performance** (e.g., BlazeFace is lightweight). | Requires **large and diverse datasets** for training. Can be computationally expensive on limited hardware. |
| **MediaPipe FaceMesh** | 3D Landmark Detection | **Precise 3D landmarking** (468 points). Detects landmarks independently per frame, making it robust to head pose/rotation. Excellent for defining complex ROI shapes. | Need for training on diverse data remains a challenge. |
| **Facial Landmarking Algorithms** (e.g., DRMF, AAM) | ROI Definition (Shape/Key Points) | Excellent for defining **unevenly shaped or multiple ROIs**. Can exclude non-skin areas (e.g., eyes/hair) for a **precise ROI**. | Localization is challenging due to the inherent complexity of **3D facial structure**. |
| **FaceBoxes**  | used for the initial face detection in each RGB video frame. It provides the bounding box for the face. | high-performance model designed to be CPU real-time with high accuracy.|  |
| **LUVLi face alignment**  |generates 68 landmark points on the face. | |  |