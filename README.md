# LabGraphics-CoHind

## OCR + LLM Post-Processing for Historical Newspapers

A pipeline for OCR text recognition with LLM-based post-processing to improve accuracy on historical Indian-English newspapers. This project combines PaddleOCR for initial text extraction with Google Gemini for intelligent error correction.

## Overview

This repository provides tools to:
- Extract text from historical newspaper images using PaddleOCR
- Apply LLM-based post-correction to fix common OCR errors
- Evaluate performance against ground truth using CER and WER metrics
- Generate visual overlays showing corrected text
- Output detailed evaluation reports in CSV and JSON formats

## Features

- **Multiple Inference Models**: Supports various fine-tuned PaddleOCR models (recommended: `mobile_3333_inference`)
- **Preprocessing**: Grayscale conversion and denoising for better OCR accuracy
- **OCR Engine**: PaddleOCR with custom fine-tuned models
- **Intelligent Post-Correction**: Uses Google Gemini to fix:
  - Character confusions
  - Malformed dates and numbers
  - OCR stutter and duplications
- **Preservation**: Maintains historical spelling, abbreviations, and formatting
- **Evaluation Metrics**: Character Error Rate (CER) and Word Error Rate (WER)
- **Visualization**: Generates overlay images with corrected text

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- PaddleOCR
- Google Gemini API key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Saransh2501/LabGraphics-CoHind.git
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API key**

Set your Gemini API key as an environment variable (recommended):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or edit `evaluation_post.py` and set the `API_KEY` variable directly.

4. **Download or place your OCR models**

Place fine-tuned PaddleOCR models in the `models/` directory. 

## Usage

### Running the Script

1. **Prepare your data**
   - Place newspaper images in `images/` directory (`.png`, `.jpg`, or `.jpeg`)
   - Place corresponding ground truth files in `ground_truth/` with the same filename but `.txt` extension

2. **Configure paths** (in `evaluation_post.py`)
```python
BASE_DIR = Path(r"")  # Project root
IMAGES_DIR = BASE_DIR / "images"
GT_DIR = BASE_DIR / "ground_truth"
MODEL_DIR = Path(r"models/mobile_3333_inference")  # Recommended model
```

3. **Run the evaluation**
```bash
python main.py
```

### Output Files

After running, you'll find:

- **`Evaluation_Reports/Evaluation_Report.csv`**: Summary table with CER/WER metrics
- **`Evaluation_Reports/json/*.json`**: Per-image results including bounding boxes and text
- **`Evaluation_Reports/images/*.png`**: Visual overlays with corrected text

## Models

### Recommended Model
**`mobile_3333_inference`** - Optimized for historical newspaper text with the best balance of accuracy and speed.

### Using Different Models

To use a different model, change the `MODEL_DIR` path in `evaluation_post.py`:
```python
MODEL_DIR = Path(r"models/your_model_name")
```

## Training Data Preparation

Use `combine_datasets.py` to create augmented training datasets:

```bash
python combine_datasets.py
```

This script:
- Combines domain-specific newspaper text with generic corpora
- Adds number-focused examples for better digit recognition
- Creates non-overlapping train/validation splits
- Applies configurable augmentation factors


