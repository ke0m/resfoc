import torch
import torch.nn as nn

class bal_ce(nn.Module):

  def __init__(self,device):
    super().__init__()
    self.__device = device

  def forward(self, logits, target):
    # Get the weights
    count_neg = (1-target).sum()
    count_pos = (target).sum()
    beta = count_neg / (count_neg + count_pos)

    # Evaluate the loss on each example in the batch
    bwgt = torch.ones(logits.size()[0])*(1-beta)
    tbceloss = nn.BCEWithLogitsLoss(reduction='none',pos_weight=beta/(1-beta))
    cost = tbceloss(logits,target)
    cost = (cost*(1-beta)).mean()

    # Return 0 if all zeros, otherwise return the loss
    zero = torch.zeros([1],dtype=torch.float).to(self.__device)
    cost = torch.where((count_pos == zero),zero,cost)

    return cost
