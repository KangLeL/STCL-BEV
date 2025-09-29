import torch


class IDManager():
    def __init__(self, history_size=1, momentum=0.8):
        super().__init__()
        # 使用字典存储每个ID的最新特征
        self.id_feature_bank = {}  # {id: feature_vector}
        self.id_age = {}  # {id: age_in_frames}
        self.momentum = momentum
        self.history_size = history_size


    def reset(self):
        """重置特征库"""
        self.id_feature_bank = {}
        self.id_age = {}


    def update_feature_bank(self, feats, targets):
        """基于ID更新特征库，只更新当前帧中出现的ID"""

        device = feats.device

        # 将特征和ID转换为CPU（避免GPU内存问题）
        feats = feats.detach().cpu()
        targets = targets.detach().cpu()
        self.id_age = {_id: age + 1 for _id, age in self.id_age.items()}  # 增加所有ID的年龄
        for feat, id in zip(feats, targets):
            id = id.item()  # 转换为标量

            if id in self.id_feature_bank:
                # 动量更新：只更新相同ID的特征
                self.id_feature_bank[id] = (
                        self.momentum * self.id_feature_bank[id] +
                        (1 - self.momentum) * feat
                )
                self.id_age[id] = 0  # 重置年龄
            else:
                # 新ID：直接存储
                self.id_feature_bank[id] = feat
                self.id_age[id] = 0

        # 可选：清理长时间未出现的ID
        self.id_feature_bank = {_id: feat for _id, feat in self.id_feature_bank.items() if
                                self.id_age[_id] <= self.history_size}
        self.id_age = {_id: age for _id, age in self.id_age.items() if age <= self.history_size}

    def get_feature_bank(self):
        """获取当前的特征库和对应ID"""
        if not self.id_feature_bank:
            return torch.empty((0, 128)), torch.empty((0,), dtype=torch.long)
        ids = list(self.id_feature_bank.keys())
        features = torch.stack([self.id_feature_bank[id] for id in ids])
        return features, torch.tensor(ids, dtype=torch.long)


