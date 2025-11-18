import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: Model training.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    print("[pt4_train] Placeholder CLI.")
    print("config    :", args.config)
    print("data-dir  :", args.data_dir)
    print("output-dir:", args.output_dir)
