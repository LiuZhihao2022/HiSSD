import wandb
import numpy as np
from collections import defaultdict

class WandbLogger:
    """
    用于记录训练信息到wandb的工具类
    """
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        self.step_metrics = defaultdict(list)  # 临时存储每个步骤的指标
        self.episode_metrics = defaultdict(list)  # 临时存储每个episode的指标
    
    def log(self, metrics, step=None, commit=True):
        """
        记录指标到wandb
        
        Args:
            metrics: 字典，包含要记录的指标
            step: 可选，当前步数
            commit: 是否立即提交到wandb
        """
        if not self.use_wandb:
            return
        
        # 确保不记录None值
        clean_metrics = {k: v for k, v in metrics.items() if v is not None}
        
        # 移除可能的torch张量
        for k, v in clean_metrics.items():
            if hasattr(v, 'item'):
                clean_metrics[k] = v.item()
        
        # 使用wandb记录
        wandb.log(clean_metrics, step=step, commit=commit)
    
    def add_step_metrics(self, metrics):
        """添加每步指标到临时存储"""
        if not self.use_wandb:
            return
            
        for k, v in metrics.items():
            if v is not None:
                if hasattr(v, 'item'):
                    v = v.item()
                self.step_metrics[k].append(v)
    
    def add_episode_metrics(self, metrics):
        """添加每episode指标到临时存储"""
        if not self.use_wandb:
            return
            
        for k, v in metrics.items():
            if v is not None:
                if hasattr(v, 'item'):
                    v = v.item()
                self.episode_metrics[k].append(v)
    
    def log_step_summaries(self, step=None, commit=True):
        """记录每步指标的平均值等统计信息"""
        if not self.use_wandb or not self.step_metrics:
            return
            
        summary_metrics = {}
        for k, values in self.step_metrics.items():
            if values:
                summary_metrics[f"{k}/mean"] = np.mean(values)
                summary_metrics[f"{k}/std"] = np.std(values)
                summary_metrics[f"{k}/min"] = np.min(values)
                summary_metrics[f"{k}/max"] = np.max(values)
        
        self.log(summary_metrics, step=step, commit=commit)
        self.step_metrics.clear()
    
    def log_episode_summaries(self, step=None, commit=True):
        """记录每episode指标的平均值等统计信息"""
        if not self.use_wandb or not self.episode_metrics:
            return
            
        summary_metrics = {}
        for k, values in self.episode_metrics.items():
            if values:
                summary_metrics[f"{k}/mean"] = np.mean(values)
                summary_metrics[f"{k}/std"] = np.std(values)
                summary_metrics[f"{k}/min"] = np.min(values)
                summary_metrics[f"{k}/max"] = np.max(values)
        
        self.log(summary_metrics, step=step, commit=commit)
        self.episode_metrics.clear()

    def watch_model(self, model, log="all", log_freq=100):
        """监控模型参数和梯度"""
        if self.use_wandb:
            wandb.watch(model, log=log, log_freq=log_freq)
