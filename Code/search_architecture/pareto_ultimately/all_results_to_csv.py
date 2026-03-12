import os
import json
import torch
import pandas as pd
from tqdm import tqdm

# --- 模块导入 ---
# [注意!] 您可能需要根据您项目的文件结构，调整下面两行的导入路径
from Code.search_architecture.lstm_search.auto_model import FlexibleGNNModel
from Code.search_architecture.random_search.run_search import load_and_split_data



def count_parameters(model: torch.nn.Module) -> int:

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    results_directory = 'Code/search_architecture/results'

    print(f"Will scan all .json files in this directory: {results_directory}")
    if not os.path.isdir(results_directory):
        print(f"[Error] Directory not found: {results_directory}")
        return

    print("Loading dataset to obtain metadata...")
    db_name = 'e'
    data, train_data, _, _ = load_and_split_data(db_name)
    if train_data is None:
        print("[Error] Data loading failed, unable to proceed.")
        return
    print("Data loaded successfully.")

    json_files_to_process = []
    for root, _, files in os.walk(results_directory):
        for file in files:
            if file.startswith(db_name) and file.endswith('.json'):
                print(f"✅ Selected: {file}")
                json_files_to_process.append(os.path.join(root, file))

    if not json_files_to_process:
        print(f"[Error] No .json files found in directory {results_directory}.")
        return

    print(f"Found {len(json_files_to_process)} result files, preparing to process...")

    all_architectures_data = []
    for file_path in tqdm(json_files_to_process, desc="Processing JSON files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)

        for result in results_data:
            architecture_config = result.get('architecture')
            if not architecture_config:
                continue

            is_valid = all(
                layer_cfg.get(key) is not None for layer_cfg in architecture_config for key in ['conv', 'act', 'norm'])
            if not is_valid:
                continue

            try:
                model = FlexibleGNNModel(architecture_config, train_data.metadata()).to('cpu')

                with torch.no_grad():
                    model.eval()
                    dummy_edge_label_index = train_data['drug', 'protein'].edge_label_index[:, :1]
                    _ = model(train_data.x_dict, train_data.edge_index_dict, dummy_edge_label_index)

                total_params = count_parameters(model)

                processed_data = {
                    'combined_reward': result.get('combined_reward', -1.0),
                    'parameters_M': total_params / 1_000_000,
                    'architecture': str(architecture_config),
                    'source_file': os.path.basename(file_path)
                }
                all_architectures_data.append(processed_data)

            except Exception:

                continue

    if not all_architectures_data:
        print("\nFailed to process any valid architectural data, unable to generate CSV file.")
        return

    results_df = pd.DataFrame(all_architectures_data)

    results_df = results_df.sort_values(by='combined_reward', ascending=False).reset_index(drop=True)

    output_csv_path = os.path.join(results_directory,f'{db_name}_results.csv')
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n--- All files processed ---")
    print(f"Integrated and saved {len(results_df)} valid architecture records to: {output_csv_path}")
    print("\nCSV file preview:")
    print(results_df.head())


if __name__ == "__main__":
    main()