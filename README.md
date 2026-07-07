# Custom Single-User Face Authentication System

This repository implements a production-ready, custom single-user face authentication system built from mathematical first principles using **OpenCV**, **NumPy**, and **SciPy**, running a **FastAPI** backend interface.

---

## 🔬 Mathematical Biometrics Pipeline

To satisfy the constraint of **no pretrained neural network or cascade models**, the entire pipeline is built on classical computer vision methods:

1. **Face Detection (`myface/detection`):** Localizes the face using a Skin-independent **Bilateral Symmetry Map** computed on image gradients, verified by a multi-scale **HOG Mean-Face Template** and eye-nose-mouth vertical gradient projection profiles.
2. **Alignment (`myface/alignment`):** Detects eye center locations using **Personalized Eye HOG Templates** registered during enrollment, aligned using standard 2D Affine rotations.
3. **Feature Extraction (`myface/feature_extraction`):** Extract normalized spatial LBP (Local Binary Patterns) histograms and HOG descriptors, projected onto a localized user-specific **PCA Eigenspace** (Eigenfaces) generated during enrollment via Singular Value Decomposition (SVD).
4. **Passive Anti-Spoofing (`myface/anti_spoof`):** Operates 3D **Shape-from-Shading (SfS)** depth estimation modeling a Lambertian surface, **2D FFT Moiré Pattern Peak Detection** scanning for screens, and chrominance diffusion rate calculations.
5. **Similarity Matching (`myface/matching`):** Computes **Mahalanobis Distance** inside the projected Eigenspace.

---

## 📂 Project Structure

```
d:/myFace/
├── requirements.txt           # Main dependencies (FastAPI, OpenCV headless, NumPy, SciPy)
├── README.md                  # System documentation
├── .env                       # Environment variables
├── myface/                    # Application package
│   ├── configuration/         # Config loader settings
│   ├── camera/                # Multi-threaded camera acquisition
│   ├── dataset/               # Enrollment image management
│   ├── preprocessing/         # CLAHE and bilateral filtering
│   ├── detection/             # Bilateral symmetry and gradient maps
│   ├── alignment/             # Eye template alignment
│   ├── feature_extraction/    # Spatial LBP/HOG + PCA projector
│   ├── anti_spoof/            # SfS + FFT Moiré liveness checks
│   ├── recognition/           # Biometric lifecycle orchestrator
│   ├── matching/              # Mahalanobis Eigenspace matching
│   ├── training/              # Centroid aggregation & SVD solver
│   ├── evaluation/            # Biometric validation reports (FAR, FRR, EER)
│   ├── database/              # File system JSON template store
│   ├── performance/           # Latency metrics profiler
│   ├── api/                   # FastAPI routing, app setup & requests
│   └── utils/                 # Logging & helper tools
└── tests/                     # Unit & integration testing
```

---

## 🚀 Setup & Installation

1. **Install Python 3.10+** (Windows, Linux, or macOS).
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment:** Customize variables in `.env` or use default values.
4. **Run Application:**
   ```bash
   uvicorn myface.api.main:app --reload
   ```

---

## 🧪 Testing

Run pytest suite to execute all unit and integration tests:
```bash
pytest -v
```
