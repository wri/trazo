import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Step 3: Fine tuning.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    print("[pt3_finetune] Placeholder CLI.")
    print("config    :", args.config)
    print("checkpoint:", args.checkpoint)
    print("data-dir  :", args.data_dir)
