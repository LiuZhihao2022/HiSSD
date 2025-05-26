import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_all_offline_runs(root_dir):
    """查找所有offline-run-*目录，返回路径列表"""
    run_paths = []
    for run_dir in os.listdir(root_dir):
        if run_dir.startswith("offline-run-"):
            run_path = os.path.join(root_dir, run_dir)
            if os.path.isdir(run_path):
                run_paths.append(run_path)
    return run_paths

def upload_run(run_path):
    print(f"Uploading wandb run: {run_path}")
    result = subprocess.run([sys.executable, "-m", "wandb", "sync", run_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Uploaded: {run_path}")
    else:
        print(f"Failed: {run_path}\n{result.stderr}")

def upload_all_offline_runs_parallel(root_dir, max_workers=9):
    run_paths = find_all_offline_runs(root_dir)
    print(f"Found {len(run_paths)} offline runs to upload.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_run, run_path) for run_path in run_paths]
        for future in as_completed(futures):
            future.result()
    print("All offline runs uploaded.")

if __name__ == "__main__":
    # offline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mcts_wandb_offline_run"))
    offline_root = "/home/lzh/HiSSD/mcts_wandb_offline_run"
    upload_all_offline_runs_parallel(offline_root, max_workers=10)
# This script is used to upload all offline runs in the specified directory to Weights & Biases (wandb).