import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os

def extract_tensorboard_data(log_dir):
    # Find the tfevents file in the directory
    event_file = [f for f in os.listdir(log_dir) if "tfevents" in f][0]
    path = os.path.join(log_dir, event_file)

    # Initialize the accumulator
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    # Extract specific metrics recorded by your RewardShapingCallback
    metrics = ["env/reward", "env/cumulative_vetoes", "env/balance"]
    data = {}

    for metric in metrics:
        if metric in ea.Tags()['scalars']:
            events = ea.Scalars(metric)
            data[metric] = [e.value for e in events]
            data[f"{metric}_step"] = [e.step for e in events]

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Point this to your latest PPO run folder
    log_folder = "./logs/margin_guard_v3/PPO_6" 
    df = extract_tensorboard_data(log_folder)
    
    # Show the last 10 steps of training
    print(df.tail(10))
    
    # Save to CSV for your report
    df.to_csv("training_history_audit.csv", index=False)