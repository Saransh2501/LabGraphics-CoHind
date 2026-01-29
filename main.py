import os
import glob
import json
from pathlib import Path
import pandas as pd
from paddleocr import PaddleOCR
from jiwer import cer, wer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
import cv2  # Added for preprocessing

# ===================== PATHS (RELATIVE) =====================
BASE_DIR = Path(r"")
IMAGES_DIR = BASE_DIR / "images"
GT_DIR = BASE_DIR / "ground_truth"
MODEL_DIR = Path(r"")
OUT_DIR = BASE_DIR / "Evaluation_Reports"
OUT_JSON_DIR = OUT_DIR / "json"
OUT_IMG_DIR = OUT_DIR / "images"
REPORT_CSV = OUT_DIR / "Evaluation_Report.csv"

# Font for overlay (Windows default). Change if needed.
FONT_PATH = r"C:\Windows\Fonts\times.ttf"
SHOW_BOXES = True
BOX_COLOR = (200, 200, 200)
CANVAS_COLOR = (255, 255, 255)

# ============================================================
# ===================== GEMINI CONFIG =========================
API_KEY = ""  # preferred over hard-coding
GEMINI_MODEL = "gemini-2.5-flash"
SYSTEM_RULES = """You are an OCR post-correction engine for historical newspapers.

STRICT RULES (never do these):
- Do NOT modernize spelling (historical spellings are allowed).
- Do NOT expand abbreviations.
- Do NOT rewrite grammar or phrasing.
- Do NOT normalize capitalization.

YOU MUST DO:
- Actively search for OCR character confusions: 0/O, 1/l/I, 5/S, 2/Z, 8/B, rn↔m, cl↔d, vv↔w, u↔n, t↔f.
- Fix malformed years/dates/numbers when the correction is obvious (example: '178o' -> '1780').
- Fix broken punctuation clearly caused by OCR.
- Remove duplicated characters caused by OCR stutter (example: 'tthe' -> 'the').

IMPORTANT:
- If you leave a line unchanged, it means you VERIFIED it has no OCR error.
- Do NOT blindly copy input.

Output valid JSON only.
"""

# ============================================================

def preprocess_newspaper(image_path):
    """
    Preprocess newspaper image: grayscale conversion, denoising, and RGB conversion for PaddleOCR.
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Light denoising (important before further processing)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # PaddleOCR expects 3-channel images
    preprocessed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return preprocessed_img


def read_lines(txt_path: str) -> list[str] | None:
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def extract_rec_texts_and_polys(pred_gen):
    """
    Reuses the same 'rec_texts' extraction idea from your eval.py (supports multiple JSON layouts).
    """
    rec_texts = None
    rec_polys = None
    for res in pred_gen:
        if not hasattr(res, "json"):
            continue
        data = res.json
        obj = None
        if isinstance(data, dict):
            obj = data
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            obj = data[0]
        if not obj:
            continue
        if "rec_texts" in obj:
            rec_texts = obj.get("rec_texts")
            rec_polys = obj.get("rec_polys")
        elif "res" in obj and isinstance(obj["res"], dict) and "rec_texts" in obj["res"]:
            rec_texts = obj["res"].get("rec_texts")
            rec_polys = obj["res"].get("rec_polys")
    if rec_texts is None:
        rec_texts = []
    if rec_polys is None:
        rec_polys = []
    return rec_texts, rec_polys


def correct_rec_texts_with_gemini(rec_texts: list[str]) -> list[str]:
    if not API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY environment variable (recommended) or fill API_KEY.")
    
    client = genai.Client(api_key=API_KEY)
    response_schema = {
        "type": "OBJECT",
        "properties": {"rec_texts": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["rec_texts"],
    }
    
    user_prompt = (
        "Correct OCR errors in these lines under the system rules.\n"
        "Return the full list with the same length and same order.\n\n"
        f"rec_texts:\n{json.dumps(rec_texts, ensure_ascii=False, indent=2)}"
    )
    
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_RULES,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )
    
    data = json.loads(resp.text)
    out = data["rec_texts"]
    if len(out) != len(rec_texts):
        raise ValueError(f"Gemini returned {len(out)} lines, expected {len(rec_texts)}.")
    return out


def quad_to_aabb(quad):
    pts = np.array(quad, dtype=np.int32)
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())


def fit_font_size(draw, text, font_path, box_w, box_h, max_size=24):
    size = max_size
    while size > 5:
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= box_w and h <= box_h:
            return font
        size -= 1
    return ImageFont.truetype(font_path, size)


def render_on_blank(img_path, rec_polys, rec_texts, out_path, font_path):
    with Image.open(img_path) as temp_img:
        width, height = temp_img.size
    pil_img = Image.new("RGB", (width, height), color=CANVAS_COLOR)
    draw = ImageDraw.Draw(pil_img)
    for quad, text in zip(rec_polys, rec_texts):
        if not str(text).strip():
            continue
        x1, y1, x2, y2 = quad_to_aabb(quad)
        box_w, box_h = x2 - x1, y2 - y1
        font = fit_font_size(draw, text, font_path, box_w * 0.95, box_h * 0.95)
        t_bbox = draw.textbbox((0, 0), text, font=font)
        t_w, t_h = t_bbox[2] - t_bbox[0], t_bbox[3] - t_bbox[1]
        text_x = x1 + (box_w - t_w) / 2 - t_bbox[0]
        text_y = y1 + (box_h - t_h) / 2 - t_bbox[1]
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
        if SHOW_BOXES:
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=1)
    pil_img.save(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    
    image_files = (
        glob.glob(os.path.join(IMAGES_DIR, "*.png"))
        + glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
    )
    
    if not image_files:
        raise SystemExit(f"No images found in {IMAGES_DIR}")
    
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_recognition_model_dir=MODEL_DIR,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    
    rows = []
    skipped_no_gt = 0
    
    for img_path in sorted(image_files):
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]
        gt_path = os.path.join(GT_DIR, f"{base}.txt")
        gt_lines = read_lines(gt_path)
        if gt_lines is None:
            skipped_no_gt += 1
            continue
        
        # Apply preprocessing
        preprocessed_img = preprocess_newspaper(img_path)
        
        # Run OCR on preprocessed image
        pred_gen = ocr.predict(preprocessed_img)
        rec_texts, rec_polys = extract_rec_texts_and_polys(pred_gen)
        rec_texts_corrected = correct_rec_texts_with_gemini(rec_texts)
        
        gt_text = "\n".join(gt_lines).strip()
        pred_text_raw = "\n".join(rec_texts).strip()
        pred_text_corr = "\n".join(rec_texts_corrected).strip()
        
        cer_raw = cer(gt_text, pred_text_raw) if pred_text_raw else 1.0
        wer_raw = wer(gt_text, pred_text_raw) if pred_text_raw else 1.0
        cer_corr = cer(gt_text, pred_text_corr) if pred_text_corr else 1.0
        wer_corr = wer(gt_text, pred_text_corr) if pred_text_corr else 1.0
        
        # Save JSON for auditability + downstream usage
        out_json = os.path.join(OUT_JSON_DIR, f"{base}.3323.corrected.json")
        obj = {
            "image": fname,
            "model": "mobile_3323_inference",
            "rec_polys": rec_polys,
            "rec_texts_original": rec_texts,
            "rec_texts": rec_texts_corrected,
        }
        Path(out_json).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        
        # Render corrected overlay image
        out_img = os.path.join(OUT_IMG_DIR, f"{base}.3323.corrected_overlay.png")
        render_on_blank(img_path, rec_polys, rec_texts_corrected, out_img, FONT_PATH)
        
        rows.append({
            "Model": "mobile_3323_inference",
            "Image": fname,
            "CER_raw": cer_raw,
            "WER_raw": wer_raw,
            "CER_corrected": cer_corr,
            "WER_corrected": wer_corr,
            "JSON": out_json,
            "OverlayImage": out_img,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_CSV, index=False)
    
    print("Evaluated images:", len(df))
    print("Skipped (no GT):", skipped_no_gt)
    if not df.empty:
        print("Median CER raw:", float(df["CER_raw"].median()))
        print("Median CER corrected:", float(df["CER_corrected"].median()))
    print("Wrote:", REPORT_CSV)


if __name__ == "__main__":
    main()
