import torch 
from torch.utils.data import Dataset
from pathlib import Path


class KDDataset(Dataset):

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        name: str = "kdd_train",
        labels_dict: dict= None
    ):
        super(KDDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.labels_dict = labels_dict
        self.name = name 

    def save(self, save_dir: str):
        save_dir = Path(save_dir)

        if not save_dir.exists():
            raise ValueError(f"Directory {save_dir} does not exist")

        raws = save_dir / "raws"

        if not raws.exists():
            raws.mkdir()
        
        file_path = raws / f"{self.name}.pt"
        torch.save((
            self.features, self.labels, self.name, self.labels_dict
        ), file_path)

    @staticmethod
    def load(saved_dir, is_train = True): 
        name = "kdd_train" if is_train else "kdd_test"

        saved_dir = Path(saved_dir)
        if not saved_dir.exists():
            raise ValueError(f"Directory {saved_dir} does not exist")
        raws = saved_dir / "raws"
        if not raws.exists():
            raise ValueError(f"Directory {raws} does not exist")
        file_path = raws / f"{name}.pt"

        features, labels, name, labels_dict = torch.load(file_path, weights_only=True)
        return KDDataset(features, labels, name, labels_dict)

    @staticmethod
    def load_all(saved_dir):
        train = KDDataset.load(saved_dir, is_train=True)
        test = KDDataset.load(saved_dir, is_train=False)
        return train, test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return (self.features[idx], self.labels[idx])