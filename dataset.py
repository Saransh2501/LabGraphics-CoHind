import os
import random

# ===================== CONFIGURATION =====================
# Paths to input label files
NEWS_TRAIN_FILE = "data/news_train.txt"
NEWS_VAL_FILE = "data/news_val.txt"
GENERIC_FILE = "data/generic_corpus.txt"
NUMBERS_FILE = "data/numbers_corpus.txt"

# Output paths
OUTPUT_TRAIN = "output/combined_train.txt"
OUTPUT_VAL = "output/combined_val.txt"

# Augmentation factors (multiplier relative to news dataset size)
GENERIC_FACTOR_TRAIN = 4.0
NUMBERS_FACTOR_TRAIN = 4.0
GENERIC_FACTOR_VAL = 3.0
NUMBERS_FACTOR_VAL = 4.0

# Random seed for reproducibility (set to None for random behavior)
RANDOM_SEED = 42
# =========================================================


def read_label_file(path):
    """Read label file and return list of non-empty lines."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def main():
    # Set random seed for reproducibility
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)
    
    # Read all label files
    print("Reading input files...")
    news_train = read_label_file(NEWS_TRAIN_FILE)
    news_val = read_label_file(NEWS_VAL_FILE)
    generic = read_label_file(GENERIC_FILE)
    numbers = read_label_file(NUMBERS_FILE)
    
    print(f"News train: {len(news_train):,}")
    print(f"News val:   {len(news_val):,}")
    print(f"Generic:    {len(generic):,}")
    print(f"Numbers:    {len(numbers):,}")
    
    # Shuffle auxiliary datasets
    random.shuffle(generic)
    random.shuffle(numbers)
    
    # ---- BUILD TRAINING SET ----
    t_gen = min(len(generic), int(len(news_train) * GENERIC_FACTOR_TRAIN))
    t_num = min(len(numbers), int(len(news_train) * NUMBERS_FACTOR_TRAIN))
    
    gen_train = generic[:t_gen]
    num_train = numbers[:t_num]
    
    print(f"\nTrain: adding {t_gen:,} generic + {t_num:,} numbers")
    
    combined_train = news_train + gen_train + num_train
    random.shuffle(combined_train)
    
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_train) + "\n")
    
    # ---- BUILD VALIDATION SET ----
    start_g = t_gen
    start_n = t_num
    
    v_gen = min(len(generic) - start_g, int(len(news_val) * GENERIC_FACTOR_VAL))
    v_num = min(len(numbers) - start_n, int(len(news_val) * NUMBERS_FACTOR_VAL))
    
    gen_val = generic[start_g:start_g + v_gen]
    num_val = numbers[start_n:start_n + v_num]
    
    print(f"Val:   adding {v_gen:,} generic + {v_num:,} numbers")
    
    combined_val = news_val + gen_val + num_val
    random.shuffle(combined_val)
    
    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_val) + "\n")
    
    # Summary
    print("\n" + "="*50)
    print("✓ Dataset combination complete!")
    print(f"  Training:   {len(combined_train):,} lines → {OUTPUT_TRAIN}")
    print(f"  Validation: {len(combined_val):,} lines → {OUTPUT_VAL}")
    print("="*50)


if __name__ == "__main__":
    main()
