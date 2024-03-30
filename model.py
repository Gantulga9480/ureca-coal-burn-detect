import torch
import torch.nn as nn
from torchmetrics.functional.classification import binary_accuracy, binary_precision, binary_recall, binary_f1_score

LEARNING_RATE = 0.0003

class Model(nn.Module):

  def __init__(self, mode='inference') -> None:
      super().__init__()
      self.mode = mode
      # Network
      self.rnn = nn.LSTM(input_size=3, hidden_size=16, num_layers=3, bias=False, batch_first=True)
      self.linear1 = nn.Linear(16, 8, bias=True)
      self.linear2 = nn.Linear(8, 1, bias=True)
      self.norm = nn.Sigmoid()
      # Optimizer
      self.loss_fn = nn.BCELoss()
      self.optim = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

  def forward(self, x):
      x = self.rnn(x)[0][:,-1]
      x = self.linear1(x)
      x = self.linear2(x)
      out = self.norm(x)
      return out

  @torch.no_grad()
  def validate(self, dataloader):
    self.eval()
    x, y = next(iter(dataloader))
    preds = self(x)
    accuracy = binary_accuracy(preds.cpu(), y.cpu())
    precision = binary_precision(preds.cpu(), y.cpu())
    recall = binary_recall(preds.cpu(), y.cpu())
    f1 = binary_f1_score(preds.cpu(), y.cpu())
    if self.mode == 'train':
      loss = self.loss_fn(preds, y).cpu().item()
      self.train()
    else:
       loss = 0
    return loss, accuracy, precision, recall, f1