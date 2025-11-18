import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Step 5: Inference and post processing.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    print("[pt5_inference] Placeholder CLI.")
    print("config    :", args.config)
    print("checkpoint:", args.checkpoint)
    print("input-dir :", args.input_dir)
    print("output-dir:", args.output_dir)
