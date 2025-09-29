from typing import Iterator
import torch
from torch.utils.data import RandomSampler
import random

class RandomPairSampler(RandomSampler):
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.arange(0, n, dtype=torch.long).view(-1, 2)[
                    torch.randperm(n // 2, generator=generator)].view(-1).tolist()
            yield from torch.arange(0, n, dtype=torch.long).view(-1, 2)[
                           torch.randperm(n // 2, generator=generator)].view(-1).tolist()[:self.num_samples % n]


class GroupSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_groups=16):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.num_groups = num_groups

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # 创建索引列表 [0, 1, 2, ..., num_samples-1]
        indices = list(range(self.num_samples))

        # 生成随机组大小（总和等于样本总数）
        group_sizes = self._generate_random_group_sizes()

        # 分组（保持原始顺序）
        groups = []
        start = 0
        for size in group_sizes:
            group = indices[start:start + size]
            groups.append(group)
            start += size

        # 打乱组间顺序
        random.shuffle(groups)

        # 将打乱后的组连接成最终序列
        for group in groups:
            yield from group  # 按顺序生成组内每个索引

    def _generate_random_group_sizes(self):
        """生成随机组大小，总和等于样本总数"""
        # 生成num_groups-1个随机切割点
        cut_points = sorted(random.sample(range(1, self.num_samples), self.num_groups - 1))

        # 计算每组大小
        sizes = []
        start = 0
        for point in cut_points:
            sizes.append(point - start)
            start = point
        sizes.append(self.num_samples - start)  # 最后一组

        # 确保所有组大小至少为1
        if any(size == 0 for size in sizes):
            return self._generate_random_group_sizes()  # 递归直到所有组大小有效

        return sizes

