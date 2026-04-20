import torch
from torch.utils.data import DataLoader, Dataset


class RegressionDataset(Dataset):
    def __init__(self, X, y_depths, y_overflow, flood_mask):
        self.X = torch.FloatTensor(X)
        self.y_depths = torch.FloatTensor(y_depths)
        self.y_overflow = torch.FloatTensor(y_overflow)
        self.flood_mask = torch.FloatTensor(flood_mask)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "y_depths": self.y_depths[idx],
            "y_overflow": self.y_overflow[idx],
            "flood_mask": self.flood_mask[idx],
        }


def get_dataloaders(train_data, val_data, test_data, batch_size=512, num_workers=0):
    train_loader = DataLoader(
        RegressionDataset(**train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        RegressionDataset(**val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        RegressionDataset(**test_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def get_test_dataloader(test_data, batch_size=512, num_workers=0):
    return DataLoader(
        RegressionDataset(**test_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
