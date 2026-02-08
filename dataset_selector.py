
import os

def list_datasets(base_dir="."):
    datasets = {
        "Vehicle A": {"no_attack": [], "fuzzing": [], "replay": [], "combined": []},
        "Vehicle B": {"no_attack": [], "fuzzing": [], "replay": [], "combined": []},
        "Vehicle C": {"no_attack": [], "fuzzing": [], "replay": [], "combined": []},
    }

    # Helper to check for keywords in file path
    def check_keywords(path):
        if "Combined" in path:
            return "combined"
        elif "Fuzzing" in path:
            return "fuzzing"
        elif "Replay" in path:
            return "replay"
        return "no_attack"
    
    # Process "Original datasets (no attacks)"
    original_dir = os.path.join(base_dir, "Original datasets (no attacks)")
    if os.path.exists(original_dir):
        for root, _, files in os.walk(original_dir):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    if "Vehicle A" in file_path:
                        datasets["Vehicle A"]["no_attack"].append(file_path)
                    elif "Vehicle B" in file_path:
                        datasets["Vehicle B"]["no_attack"].append(file_path)
                    elif "Vehicle C" in file_path:
                        datasets["Vehicle C"]["no_attack"].append(file_path)

    # Process "Datasets with attacks"
    attacks_dir = os.path.join(base_dir, "Datasets with attacks")
    if os.path.exists(attacks_dir):
        for root, _, files in os.walk(attacks_dir):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    attack_type = check_keywords(file_path)
                    if "Vehicle A" in file_path:
                        datasets["Vehicle A"][attack_type].append(file_path)
                    elif "Vehicle B" in file_path:
                        datasets["Vehicle B"][attack_type].append(file_path)
                    elif "Vehicle C" in file_path:
                        datasets["Vehicle C"][attack_type].append(file_path)
    return datasets

def print_dataset_summary(datasets):
    print("Available CAN Bus Datasets:")
    for vehicle, scenarios in datasets.items():
        print(f"  {vehicle}:")
        for scenario, files in scenarios.items():
            if files:
                print(f"    {scenario.replace('_', ' ').title()} ({len(files)} files):")
                for i, file_path in enumerate(files):
                    print(f"      [{i+1}] {file_path}")
            else:
                print(f"    {scenario.replace('_', ' ').title()}: No files found.")

if __name__ == "__main__":
    available_datasets = list_datasets()
    print_dataset_summary(available_datasets)

    # Example of how to select a file (can be extended with user input)
    # selected_file = available_datasets["Vehicle A"]["fuzzing"][0]
    # print(f"Selected for analysis: {selected_file}")
