import os
import shutil

def find_all_wandb_runs_with_online_models(root_dir):
    """递归查找同时包含wandb和online_models文件夹的目录，返回wandb目录路径列表"""
    wandb_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "wandb" in dirnames and "online_models" in dirnames:
            wandb_dir = os.path.join(dirpath, "wandb")
            wandb_dirs.append(wandb_dir)
            # 不再递归wandb和online_models
            dirnames.remove("wandb")
            dirnames.remove("online_models")
    return wandb_dirs

def find_all_wandb_runs(root_dir):
    """递归查找包含wandb文件夹的目录，返回wandb目录路径列表"""
    wandb_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "wandb" in dirnames:
            wandb_dir = os.path.join(dirpath, "wandb")
            wandb_dirs.append(wandb_dir)
            dirnames.remove("wandb")
    return wandb_dirs

def find_all_offline_runs(root_dir):
    """查找所有offline-run-*目录，返回路径列表"""
    run_paths = []
    for run_dir in os.listdir(root_dir):
        if run_dir.startswith("offline-run-"):
            run_path = os.path.join(root_dir, run_dir)
            if os.path.isdir(run_path):
                run_paths.append(run_path)
    return run_paths

def copy_offline_runs_to_target(wandb_dirs, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for wandb_dir in wandb_dirs:
        for run_dir in os.listdir(wandb_dir):
            if run_dir.startswith("offline-run-"):
                src = os.path.join(wandb_dir, run_dir)
                if os.path.isdir(src):
                    dst = os.path.join(target_dir, run_dir)
                    # 若目标已存在则先删除再复制
                    if os.path.exists(dst):
                        print(f"Target exists: {dst}, removing old directory.")
                        shutil.rmtree(dst)
                    print(f"Copying {src} to {dst}")
                    shutil.copytree(src, dst)


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/hier_mcts/sc2"))
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mcts_wandb_offline_run"))
    # Find offline wandb dirs with online models
    wandb_dirs = find_all_wandb_runs_with_online_models(root_dir)
    # Find offline wandb dirs no matter with or without online models
    # wandb_dirs = find_all_wandb_runs(root_dir)
    print(f"Found {len(wandb_dirs)} wandb dirs with online_models.")
    copy_offline_runs_to_target(wandb_dirs, target_dir)
    print("All offline runs copied.")

if __name__ == "__main__":
    main()
