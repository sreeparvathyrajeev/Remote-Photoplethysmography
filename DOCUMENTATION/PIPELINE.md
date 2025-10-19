
**Camera: Logitech webcam V-U0018**

**Dataset: UBFC-rPPG Dataset 2**






| Model/Algorithm | Usage|  
| :--- | :--- |
| **FaceBoxes**  | used for the initial face detection in each RGB video frame. The raw input video frames are first processed using the FaceBoxes algorithm , to identify and crop the face area.
| **LUVLi face alignment**  |The cropped face image is then passed to a landmark detection module, the LUVLi landmark detection module , which generates 68 landmark points on the face. These 68 initial landmarks are then interpolated to create 77 additional landmarks , spanning the cheeks, chin, and forehead.| 