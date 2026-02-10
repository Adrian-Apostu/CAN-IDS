import os
import subprocess
import sys
from datetime import datetime
from dataset_selector import list_datasets


def run_analysis_pipeline(output_log="experiment_log.txt"):
    """
    Run comprehensive experiments on CAN bus intrusion detection datasets.
    Logs all results to a file for easy review.
    """
    all_datasets = list_datasets()

    # Create log file
    log_file = open(output_log, "w")
    log_file.write(f"CAN Bus Intrusion Detection Experiments\n")
    log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n\n")

    selected_experiments = [
        # --- Experiment 1: Fuzzing Detection on all vehicles ---
        {"vehicle": "Vehicle A", "scenario": "fuzzing", "index": 0, "models": ["LogisticRegression", "RandomForest"]},
        {"vehicle": "Vehicle B", "scenario": "fuzzing", "index": 0, "models": ["LogisticRegression", "RandomForest"]},
        {"vehicle": "Vehicle C", "scenario": "fuzzing", "index": 0, "models": ["LogisticRegression", "RandomForest"]},

        # --- Experiment 2: Replay Detection on all vehicles ---
        {"vehicle": "Vehicle A", "scenario": "replay", "index": 0, "models": ["LogisticRegression", "RandomForest"]},
        {"vehicle": "Vehicle B", "scenario": "replay", "index": 0, "models": ["LogisticRegression", "RandomForest"]},
        {"vehicle": "Vehicle C", "scenario": "replay", "index": 0, "models": ["LogisticRegression", "RandomForest"]},

        # --- Experiment 3: Combined Attack Detection on all vehicles ---
        {"vehicle": "Vehicle A", "scenario": "combined", "index": 0, "models": ["LogisticRegression", "RandomForest"]},
        {"vehicle": "Vehicle B", "scenario": "combined", "index": 0, "models": ["LogisticRegression", "RandomForest"]},
        {"vehicle": "Vehicle C", "scenario": "combined", "index": 0, "models": ["LogisticRegression", "RandomForest"]}
    ]

    total_experiments = sum(len(exp["models"]) for exp in selected_experiments)
    completed = 0
    failed = 0

    for exp_num, exp in enumerate(selected_experiments, 1):
        vehicle = exp["vehicle"]
        scenario = exp["scenario"]
        index = exp["index"]
        models = exp["models"]

        if vehicle in all_datasets and scenario in all_datasets[vehicle] and len(
                all_datasets[vehicle][scenario]) > index:
            file_path = all_datasets[vehicle][scenario][index]

            header = f"\n{'=' * 80}\n"
            header += f"Experiment Set {exp_num}/{len(selected_experiments)}: {vehicle} - {scenario.title()}\n"
            header += f"Dataset: {os.path.basename(file_path)}\n"
            header += f"{'=' * 80}\n"

            print(header)
            log_file.write(header)

            for model in models:
                completed += 1
                progress = f"[{completed}/{total_experiments}] Running {model}..."
                print(progress)
                log_file.write(progress + "\n")

                # Use analyzer_improved.py if it exists, otherwise use analyzer.py
                analyzer_script = "analyzer_improved.py" if os.path.exists("analyzer_improved.py") else "analyzer.py"

                command = [
                    sys.executable,  # Use current Python interpreter
                    analyzer_script,
                    "--dataset", file_path,
                    "--model", model,
                    "--sample_frac", "0.10"  # Adjust this as needed
                ]

                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
                    print("✓ Success\n")
                    log_file.write("✓ Success\n")
                    log_file.write(result.stdout)
                    log_file.write("\n" + "-" * 80 + "\n")

                except subprocess.TimeoutExpired:
                    failed += 1
                    error_msg = f"✗ TIMEOUT: Experiment exceeded 5 minutes\n"
                    print(error_msg)
                    log_file.write(error_msg)

                except subprocess.CalledProcessError as e:
                    failed += 1
                    error_msg = f"✗ ERROR: {e}\n{e.stderr}\n"
                    print(error_msg)
                    log_file.write(error_msg)

                except FileNotFoundError:
                    failed += 1
                    error_msg = f"✗ ERROR: {analyzer_script} not found.\n"
                    print(error_msg)
                    log_file.write(error_msg)
        else:
            warning = f"\n⚠ WARNING: Dataset not found for {vehicle}, {scenario.title()}\n"
            warning += f"Available: {all_datasets.get(vehicle, {}).get(scenario, [])}\n"
            print(warning)
            log_file.write(warning)

    # Summary
    summary = f"\n{'=' * 80}\n"
    summary += "EXPERIMENT SUMMARY\n"
    summary += f"{'=' * 80}\n"
    summary += f"Total experiments: {total_experiments}\n"
    summary += f"Completed: {completed - failed}\n"
    summary += f"Failed: {failed}\n"
    summary += f"Success rate: {((completed - failed) / total_experiments * 100):.1f}%\n"
    summary += f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    print(summary)
    log_file.write(summary)
    log_file.close()

    print(f"\nFull log saved to: {output_log}")


if __name__ == "__main__":
    run_analysis_pipeline()