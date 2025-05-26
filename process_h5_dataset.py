import os
import numpy as np

import h5py

def read_data_and_process(datapath, vlt_name, vlt_uid):
    # 1. 读取并拼接所有 h5 文件
    data = {}
    with h5py.File(datapath, 'r') as f:
        for k in f.keys():
            if k not in data:
                data[k] = f[k][:]
            else:
                data[k] = np.concatenate((data[k], f[k][:]), axis=0)

    # 2. 根据 data['actions'] 构建 one-hot 编码
    #    data['actions'] shape: [batch, episode_len, num_agents, 1]
    #    data['avail_actions'] shape: [batch, episode_len, num_agents, action_dim]
    action_dim = data['avail_actions'].shape[-1]
    # 提取整数索引
    actions = data['actions'][..., 0].astype(np.int64)
    bs, ep_len, num_agents = actions.shape

    # 初始化 one-hot 数组
    actions_onehot = np.zeros((bs, ep_len, num_agents, action_dim), dtype=np.float32)

    # 填充 one-hot
    for i in range(bs):
        for t in range(ep_len):
            for a in range(num_agents):
                act = actions[i, t, a].astype(np.int64)
                actions_onehot[i, t, a, act] = 1.0

    # 3. 整合出参，保存为新的 h5
    out_data = {}
    for k, v in data.items():
        out_data[k] = v
    out_data['actions_onehot'] = actions_onehot

    # 创建输出路径并写文件
    h5_dir = os.path.join('dataset', vlt_name, vlt_uid, 'unknown')
    os.makedirs(h5_dir, exist_ok=True)
    h5_path = os.path.join(h5_dir, f'{vlt_name}.h5')
    with h5py.File(h5_path, 'w') as f:
        for k, v in out_data.items():
            f.create_dataset(k, data=v, compression='gzip')
    print(f"保存新数据到: {h5_path}")

if __name__ == "__main__":
    # 示例用法
    read_data_and_process(
        datapath="6h_vs_8z_expert.h5",
        vlt_name="6h_vs_8z",
        vlt_uid="expert",
    )
