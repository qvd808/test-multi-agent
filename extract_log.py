import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os

def extract_tensorboard_data(log_dir):
    # Find the tfevents file in the directory
    event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        return pd.DataFrame()
    
    path = os.path.join(log_dir, event_files[0])

    # Initialize the accumulator
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    # Extract specific metrics recorded by your RewardShapingCallback
    metrics = ["env/reward", "env/cumulative_vetoes", "env/balance", "env/price", "env/action", "env/is_vetoed"]
    data = {}

    for metric in metrics:
        if metric in ea.Tags()['scalars']:
            events = ea.Scalars(metric)
            data[metric] = [e.value for e in events]
            data[f"{metric}_step"] = [e.step for e in events]

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Automatically find the latest PPO folder
    base_dir = "./logs/margin_guard_v4/"
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} not found.")
        exit(1)
        
    ppo_folders = [f for f in os.listdir(base_dir) if f.startswith("PPO_")]
    if not ppo_folders:
        print(f"Error: No PPO logs found in {base_dir}")
        exit(1)
    
    # Sort numerically by index
    ppo_folders.sort(key=lambda x: int(x.split("_")[1]))
    latest_folder = os.path.join(base_dir, ppo_folders[0])
    
    print(f"Extracting logs from: {latest_folder}")
    df = extract_tensorboard_data(latest_folder)
    
    if df.empty:
        print("Error: Extracted DataFrame is empty. Check if logged tags match and logger flushed.")
    else:
        # Show the last 10 steps of training
        print(df.tail(10))
        # Save to CSV for your report
        df.to_csv("training_history_audit.csv", index=False)