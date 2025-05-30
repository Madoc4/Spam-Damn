import sys
from train import train
from predict import run

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py train <spam_dir> <ham_dir> <model_prefix>")
        print("   or: python main.py predict <model_prefix> <email_file>")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "train":
        if len(sys.argv) != 5:
            print("Usage: python main.py train <spam_dir> <ham_dir> <model_prefix>")
            sys.exit(1)
        spam_dir = sys.argv[2]
        ham_dir = sys.argv[3]
        prefix = sys.argv[4]
        train_model(spam_dir, ham_dir, prefix)
        print("Training Completed")

    elif cmd == "predict":
        if len(sys.argv) != 4:
            print("Usage: python main.py predict <model_prefix> <email_file>")
            sys.exit(1)
        prefix = sys.argv[2]
        email_file = sys.argv[3]
        result = run(email_file, prefix)
        print(f"Prediction: {result}")
    
    else: 
        print("Use 'train' or 'predict'.")

if __name__ == "__main__":
    main()
