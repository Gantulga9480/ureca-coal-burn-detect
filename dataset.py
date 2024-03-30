import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):

  def __init__(self, raw_data, device='cpu') -> None:
      super().__init__()
      self.raw_data = raw_data
      self.device = device

  def __len__(self):
      return len(self.raw_data)

  def __getitem__(self, index):
      x = torch.tensor(self.raw_data[index, :, :-1]).float().to(self.device)
      y = torch.tensor(self.raw_data[index, :, -1][0]).float().unsqueeze(-1).to(self.device)
      return x, y