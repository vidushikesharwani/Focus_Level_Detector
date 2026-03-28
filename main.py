import os
import subprocess
import sys

def run_file(filename, description):
    print("\n" + "=" * 55)
    print(f"  {description}")
    print("=" * 55)
    result = subprocess.run(
        [sys.executable, filename],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"\n Error in {filename}. Stopping.")
        sys.exit(1)
    print(f"\n {filename} completed successfully!")

def main():
    print("\n" + "=" * 55)
    print("FOCUS LEVEL DETECTOR - FULL PIPELINE")
    print("Powered by Machine Learning")
    print("=" * 55)
    print("""
Choose what you want to do:

  1 - Run full pipeline (first time setup)
      generate - visualise - train - predict

  2 - Just predict my focus score today

  3 - View my weekly progress tracker

  4 - Retrain the model with fresh data
    """)

    choice = input("Enter your choice (1/2/3/4): ").strip()

    if choice == "1":
        # ── Full pipeline ──────────────────────────────────
        run_file("generate.py",
                 "Step 1 of 4 - Generating Dataset....")

        run_file("visualize.py",
                 "Step 2 of 4 - Creating Visualizations....")

        run_file("train_model.py",
                 "Step 3 of 4 - Training ML Models....")

        run_file("predict.py",
                 "Step 4 of 4 - Predicting Your Focus Score...")

        print("\n" + "=" * 55)
        print(" Full pipeline complete!")
        print("Check your data/ folder for all saved files.")
        print("=" * 55)

    elif choice == "2":
        # ── Just predict ───────────────────────────────────
        if not os.path.exists("data/best_model.pkl"):
            print("\n No trained model found!")
            print("   Please run option 1 first to train the model.")
        else:
            run_file("predict.py",
                     "Predicting Your Focus Score...")

    elif choice == "3":
        # ── Weekly tracker ─────────────────────────────────
        if not os.path.exists("data/weekly_tracker.csv"):
            print("\n  No tracking data found!")
            print("Run predict.py a few times first.")
        else:
            run_file("weekly_tracker.py",
                     "Loading Your Weekly Progress...")

    elif choice == "4":
        # ── Retrain only ───────────────────────────────────
        run_file("generate.py",
                 "Step 1 of 2 - Regenerating Dataset...")
        run_file("train_model.py",
                 "Step 2 of 2 - Retraining Models....")
        print("\n Model retrained and saved successfully!")

    else:
        print("\n Invalid choice. Please enter 1, 2, 3 or 4.")

if __name__ == "__main__":
    main()
