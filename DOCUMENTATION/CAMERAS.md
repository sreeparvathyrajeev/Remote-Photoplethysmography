# RPPG Camera Options: Comparison of Available Modalities

| Camera Type | Base / League | Key Advantages for rPPG | Key Limitations for rPPG |
| :--- | :--- | :--- | :--- |
| **Standard RGB Camera** (Webcam, Smartphone, etc.) | **Base (Most Common)** | **Low Cost & Ubiquitous:** Most accessible and affordable hardware, making them easy to implement and deploy in telehealth or home settings. | **High Noise Sensitivity:** Signals are easily corrupted by motion artifacts (subject movement) and illumination changes (ambient light flicker, indoor lights). **Low SNR:** The rPPG signal strength is extremely low compared to noise. |
| **Thermal Camera** (Long-wave Infrared) | **Specialized / High-End** | **Unaffected by Lighting:** Measures heat/infrared radiation, so its performance is independent of visible lighting variations. **Motion-Based Detection:** Can be used to extract physiological information (heart pulse, respiratory rate) by tracking subtle blood vessel movement due to heartbeat. | **High Cost:** Typically much more expensive than RGB cameras. **Thermal Noise:** Can pick up ambient and environmental temperatures, creating noise that impacts the acquired data. **Lower Resolution:** Often has lower resolution compared to RGB. |
| **Near-Infrared (NIR) Camera** | **Specialized / Hybrid** | **Low/No Light Operation:** Promises attractive applications in darkness or low-light conditions (e.g., driver monitoring at night). | **Lower PPG Strength:** The PPG signal strength (AC/DC) is much lower in the NIR spectrum than in the RGB spectrum, making robust monitoring challenging. **Higher Cost:** Generally more expensive than standard RGB. |
| **CCD Camera** | **Specialized / Laboratory** | Used in many research and older studies. | **Expensive & Bulky:** Not feasible to deploy on a limited budget. |

---

### Key Hardware Specifications

| Specification | Importance | Research Finding |
| :--- | :--- | :--- |
| **Frame Rate (FPS)** | **CRUCIAL** | A common range is **15 to 30 FPS** for most low-cost devices. Deep Learning models can be affected by frame rate, with some showing better performance at lower FPS (e.g., 12.9 FPS). **A stable FPS is critical**. |
| **Resolution** | **MODERATE** | Resolution can be reduced (e.g., to half size) without a major impact on HR estimation. |
| **Video Compression** | **HIGH** | Compression algorithms (like H.264, H.265) often **degrade video quality** by eliminating subtle pixel variations, which directly reduces rPPG accuracy. |