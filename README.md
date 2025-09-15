# Drowning Detection with YOLOv8 + SAM + DeepSORT (Streamlit)

A Streamlit app for **people detection and tracking in sea** to support **drowning-detection**.
It combines **YOLOv8** (Ultralytics) for detection, **DeepSORT** for multi-object tracking, and **SAM (Segment Anything)** to define a water-region mask (used here to derive a static water bounding box at start).

## Features
- 🔍 **YOLOv8** person detection
- 🧭 **DeepSORT** multi-object tracking (IDs over time)
- ✂️ **SAM (Segment Anything)**-assisted water-area selection (single-shot to compute a static water bounding box)
- 🖼️ Video input via file upload 
- 📈 Console logs for YOLO / DeepSORT stats per frame
- 🧪 Evaluated papers listed under `papers/` with notes in `evaluation/`


## Setup

### 1) Create & activate a virtual environment
```bash
# Option A: venv (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Option B: conda
conda create -n drown python=3.10 -y
conda activate drown
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


### 3) Model weights & data
- **YOLOv8:** The app can load built-in Ultralytics weights (e.g., `yolov8n.pt`) automatically.
- **SAM:** Place the SAM checkpoint (e.g., `sam_vit_h.pth`) in the project root or update the path in `app.py`:
  ```python
  sam_checkpoint = "sam_vit_h.pth"
  model_type = "vit_h"
  ```

## Run the App
```bash
streamlit run app.py
```
Then open the URL shown in the terminal (usually http://localhost:8501).

## Usage
1. Upload a video.
2. (First frames) Use SAM to define the water area once; the app derives a **static water bounding box**.
3. Run detection+tracking; persons outside the water-box are ignored if configured.
4. Inspect the on-screen overlays and console logs for frame-by-frame stats( e.g., `examples/example1.mp4`, `examples/example2.mp4`).

## Notes on Evaluation
- See `papers/` for the list of related work used in the project.
- See `papers/evaluation` for your experiment notes, metrics, and comparisons across models / configs.

## Tips
- If you see `non-fast-forward` errors when pushing to GitHub, either pull with rebase or push with `--force-with-lease` if the remote is disposable.
- Consider adding a `.gitignore` to avoid committing large files (models, videos, checkpoints).

## License
Add your license choice here (e.g., MIT).

## Acknowledgments
- **Ultralytics YOLOv8**
- **DeepSORT**
- **Meta AI Segment Anything (SAM)**
