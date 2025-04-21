import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_all_wandb_runs(root_dir):
    """递归查找所有wandb run目录，返回run_path列表"""
    run_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "wandb" in dirnames:
            wandb_dir = os.path.join(dirpath, "wandb")
            for run_dir in os.listdir(wandb_dir):
                run_path = os.path.join(wandb_dir, run_dir)
                if os.path.isdir(run_path) and run_dir.startswith("offline-run-"):
                    run_paths.append(run_path)
            dirnames.remove("wandb")
    return run_paths

def upload_run(run_path):
    print(f"Uploading wandb run: {run_path}")
    # 用python -m wandb sync，避免PATH问题
    result = subprocess.run([sys.executable, "-m", "wandb", "sync", run_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Uploaded: {run_path}")
    else:
        print(f"Failed: {run_path}\n{result.stderr}")

def upload_all_wandb_runs_parallel(root_dir, max_workers=9):
    run_paths = find_all_wandb_runs(root_dir)
    print(f"Found {len(run_paths)} wandb runs to upload.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_run, run_path) for run_path in run_paths]
        for future in as_completed(futures):
            # 触发异常打印
            future.result()
    print("All wandb runs uploaded.")

if __name__ == "__main__":
    sc2_root = os.path.join(os.path.dirname(__file__), "../results/hier_mcts/sc2")
    sc2_root = os.path.abspath(sc2_root)
    upload_all_wandb_runs_parallel(sc2_root, max_workers=9)
