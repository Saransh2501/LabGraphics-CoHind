import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from paddleocr import PaddleOCR
from jiwer import cer, wer

# ================= CONFIGURATION =================
# 1. Folder containing your Images
IMAGES_DIR = r""

# 2. Folder containing your Ground Truth .txt files (Must match image names)
GT_DIR = r""

# 3. Folder containing all your Model subfolders (inference versions!)
MODELS_ROOT = r""

# 4. Where to save graphs and CSVs
OUTPUT_REPORT_DIR = r""
# =================================================

import cv2
import numpy as np

def preprocess_newspaper(image_path):

    # Read image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light denoising (important before Otsu)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)


    # PaddleOCR expects 3-channel images
    otsu_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return otsu_rgb


def read_ground_truth(txt_path):
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def pick_rec_model_name_from_folder(folder_name: str) -> str:
    name = folder_name.lower()
    if "server" in name:
        return "PP-OCRv5_server_rec"
    if "mobile" in name:
        return "en_PP-OCRv5_mobile_rec"
    raise ValueError(f"Cannot infer rec model type from folder name: {folder_name} "
                     f"(expected 'server' or 'mobile' in name)")

def run_evaluation():
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

    # Find all inference folders
    all_subs = glob.glob(os.path.join(MODELS_ROOT, "*_inference"))
    model_paths = {os.path.basename(p): p for p in all_subs}  # keep full folder name

    if not model_paths:
        print("No '*_inference' folders found! Did you run the export script?")
        return None

    # Images
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.png")) + \
                  glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))

    results = []
    print(f"Starting Evaluation on {len(model_paths)} models and {len(image_files)} images...\n")

    for model_folder_name, model_dir in model_paths.items():
        try:
            rec_name = pick_rec_model_name_from_folder(model_folder_name)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        print(f"--- Evaluating: {model_folder_name} | rec={rec_name} ---")

        try:
            ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name=rec_name,
                text_recognition_model_dir=model_dir,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
        except Exception as e:
            print(f"[SKIP] Failed to load {model_folder_name}: {e}")
            continue

        for img_path in image_files:
            file_name = os.path.basename(img_path)
            file_base = os.path.splitext(file_name)[0]

            # Ground truth
            gt_path = os.path.join(GT_DIR, f"{file_base}.txt")
            gt_text = read_ground_truth(gt_path)
            if gt_text is None:
                continue

            try:
                # Preprocess image before OCR
                preprocessed_img = preprocess_newspaper(img_path)
                # Run OCR on preprocessed image
                pred_gen = ocr.predict(preprocessed_img)


                full_pred_text = ""
                for res in pred_gen:
                    if not hasattr(res, "json"):
                        continue
                    data = res.json

                    # Your confirmed format: top-level "rec_texts": [...]
                    lines = []
                    if isinstance(data, dict) and "rec_texts" in data:
                        lines = data["rec_texts"]
                    elif isinstance(data, dict) and "res" in data and isinstance(data["res"], dict) and "rec_texts" in data["res"]:
                        lines = data["res"]["rec_texts"]
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "rec_texts" in data[0]:
                        lines = data[0]["rec_texts"]

                    if lines:
                        full_pred_text = "\n".join(lines)

                pred_text = full_pred_text.strip()

                error_cer = cer(gt_text, pred_text) if pred_text else 1.0
                error_wer = wer(gt_text, pred_text) if pred_text else 1.0
                accuracy = max(0, 1 - error_cer)

                results.append({
                    "Model": model_folder_name,  # IMPORTANT: unique per folder
                    "RecArch": rec_name,         # server vs mobile (handy for grouping)
                    "Image": file_name,
                    "CER": error_cer,
                    "WER": error_wer,
                    "Accuracy": accuracy
                })

            except Exception as e:
                print(f"[Error] {file_name} with {model_folder_name}: {e}")

    return pd.DataFrame(results)


def compute_ranks(df):
    """
    For each image, rank models based on CER (lower = better).
    Then compute average rank and MAD per model.
    """
    ranked_df = df.copy()

    # Rank models per image
    ranked_df["Rank"] = ranked_df.groupby("Image")["CER"] \
                                  .rank(method="average", ascending=True)

    # Aggregate rank statistics per model
    rank_stats = ranked_df.groupby("Model")["Rank"].agg(
        Avg_Rank="mean",
        Median_Rank="median",
        MAD_Rank=lambda x: (x - x.median()).abs().median()
    ).sort_values("Avg_Rank")

    return ranked_df, rank_stats


def generate_visualizations(df):
    sns.set_theme(style="whitegrid")
    
    # 1. Box Plot (Consistency Check)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Model", y="CER", palette="viridis")
    plt.title("Model Consistency Analysis (Character Error Rate)")
    plt.ylabel("CER (Lower is Better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_REPORT_DIR, "1_Model_Consistency_Boxplot.png"))
    print("Graph 1 Saved: Model Consistency")

    # 2. Bar Chart (Average Performance)
    plt.figure(figsize=(12, 6))
    avg_df = df.groupby("Model")["CER"].median().reset_index().sort_values("CER")
    sns.barplot(data=avg_df, x="Model", y="CER", palette="magma")
    plt.title("Average Character Error Rate (Lower is Better)")
    plt.ylabel("Average CER")
    plt.xticks(rotation=45)
    
    # Add labels on top
    for index, row in avg_df.iterrows():
        plt.text(index, row.CER, f'{row.CER:.3f}', color='black', ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_REPORT_DIR, "2_Average_Performance_Bar.png"))
    print("Graph 2 Saved: Average Performance")

    # 3. Line Chart (Image-wise Breakdown)
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x="Image", y="CER", hue="Model", marker="o")
    plt.title("Performance per Image (Difficulty Analysis)")
    plt.xticks(rotation=90)
    plt.ylabel("CER (Lower is Better)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_REPORT_DIR, "3_Per_Image_breakdown.png"))
    print("Graph 3 Saved: Per Image Breakdown")


def generate_advanced_visualizations(df):
    sns.set_theme(style="white")
    

    pivot_df = df.pivot(index="Image", columns="Model", values="CER")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="Reds", linewidths=.5)
    plt.title("Error Heatmap: Darker Red = Worse OCR")
    plt.ylabel("Image Name")
    plt.xlabel("Model Name")
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_REPORT_DIR, "Advanced_Error_Heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Saved Heatmap: {heatmap_path}")

    
    report_path = os.path.join(OUTPUT_REPORT_DIR, "Worst_Errors_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== DEEP DIVE: WORST PREDICTIONS PER MODEL ===\n\n")
        
        for model in df["Model"].unique():
            f.write(f"--- MODEL: {model} ---\n")
            # Get top 3 worst images for this model
            worst_cases = df[df["Model"] == model].sort_values("CER", ascending=False).head(3)
            
            for _, row in worst_cases.iterrows():
                f.write(f"IMAGE: {row['Image']} (CER: {row['CER']:.2f})\n")
                
                # But typically we just flag them here.
                f.write(f"   [See ground_truth/{os.path.splitext(row['Image'])[0]}.txt]\n")
                f.write("-" * 40 + "\n")
            f.write("\n")
            
    print(f"Saved Text Report: {report_path}")
    
def generate_violin_plots(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=df,
        x="Model",
        y="CER",
        inner="quartile",
        cut=0,
        palette="viridis"
    )

    plt.title("CER Distribution per Model (Violin Plot)")
    plt.ylabel("CER (Lower is Better)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(OUTPUT_REPORT_DIR, "4_CER_Violin_Plot.png")
    plt.savefig(path)
    print(f"Saved Violin Plot: {path}")



def generate_rank_heatmap(ranked_df):
    pivot_rank = ranked_df.pivot(
        index="Image",
        columns="Model",
        values="Rank"
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pivot_rank,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5
    )

    plt.title("Per-Image Model Ranking (Lower Rank = Better)")
    plt.ylabel("Image")
    plt.xlabel("Model")
    plt.tight_layout()

    path = os.path.join(OUTPUT_REPORT_DIR, "5_Rank_Heatmap.png")
    plt.savefig(path)
    print(f"Saved Rank Heatmap: {path}")


if __name__ == "__main__":
    print("Starting Evaluation Pipeline...")
    df = run_evaluation()
    
    if df is not None and not df.empty:
        # Save full CSV
        csv_path = os.path.join(OUTPUT_REPORT_DIR, "final_evaluation_report.csv")
        df.to_csv(csv_path, index=False)

        # ---------------- RANK ANALYSIS ----------------
        ranked_df, rank_stats = compute_ranks(df)

        ranked_csv = os.path.join(OUTPUT_REPORT_DIR, "image_wise_model_ranks.csv")
        ranked_df.to_csv(ranked_csv, index=False)

        rank_stats_csv = os.path.join(OUTPUT_REPORT_DIR, "model_rank_statistics.csv")
        rank_stats.to_csv(rank_stats_csv)

        print("\n=== MODEL RANKING SUMMARY ===")
        print(rank_stats)

        # ---------------- SUMMARY STATS ----------------
        summary = df.groupby("Model")[["CER", "WER", "Accuracy"]].median().sort_values("CER")
        summary_csv_path = os.path.join(OUTPUT_REPORT_DIR, "model_summary_stats.csv")
        summary.to_csv(summary_csv_path)

        # ---------------- VISUALS ----------------
        generate_visualizations(df)
        generate_advanced_visualizations(df)
        generate_violin_plots(df)
        generate_rank_heatmap(ranked_df)

        print("\n[DONE] Check the 'Evaluation_Reports' folder.")
    else:
        print("Evaluation failed or no data found.")
