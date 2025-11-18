import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Training data preparation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    print("[pt2_dataprep] Placeholder CLI.")
    print("config    :", args.config)
    print("input-dir :", args.input_dir)
    print("output-dir:", args.output_dir)
