from typing import Iterable, Iterator, List

import torch
from torch.utils.data import IterableDataset


class Txt2ImgIterableBaseDataset(IterableDataset):
    """
    轻量版占位实现，用于判断/兼容可迭代数据集（源自 LDM 的 Dataset 基类）。
    实际数据加载逻辑应由子类实现 `__iter__` 或 `get_example`。
    """

    def __init__(self, num_records: int = 0) -> None:
        super().__init__()
        self.num_records = int(num_records) if num_records else 0
        # valid_ids 用于分片；默认使用 0..num_records-1
        self.valid_ids: List[int] = list(range(self.num_records))
        self.sample_ids: List[int] = list(self.valid_ids)

    def __len__(self) -> int:
        # DataLoader 会用到 __len__，若未知长度则返回 num_records 或 sample_ids 长度
        if self.sample_ids:
            return len(self.sample_ids)
        return self.num_records

    def __iter__(self) -> Iterator:
        # 子类应重写；这里提供一个最小可用实现，按 sample_ids 迭代并调用 get_example
        if not self.sample_ids:
            return iter(())
        return (self.get_example(i) for i in self.sample_ids)

    def get_example(self, idx: int):
        raise NotImplementedError("Txt2ImgIterableBaseDataset.get_example 需由子类实现")


# 一些代码会检查 torch.utils.data.get_worker_info().dataset 是否是该类型
__all__ = ["Txt2ImgIterableBaseDataset"]


