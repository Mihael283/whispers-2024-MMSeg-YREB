import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lovasz_losses import lovasz_softmax  # If using Lovász-Softmax

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            if self.alpha.size(0) < input.size(1):
                # If alpha has fewer elements than classes, repeat the last value
                self.alpha = torch.cat([self.alpha, self.alpha[-1].repeat(input.size(1) - self.alpha.size(0))])
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n,h,w = tensor.size()
        one_hot = torch.zeros(n,self.classes,h,w).to(tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)
        
        # Numerator Product
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

        #Denominator 
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)

        loss = inter/union

        ## Return average loss over classes and batch
        return 1-loss.mean()



class CombinedLoss(nn.Module):
    def __init__(self, weight_ce, weight_secondary=1.0, secondary_loss='miou', num_classes=10):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight_ce)
        self.weight_secondary = weight_secondary
        self.secondary_loss_type = secondary_loss.lower()
        self.num_classes = num_classes
        
        if self.secondary_loss_type == 'miou':
            self.secondary = mIoULoss(n_classes=num_classes)
        elif self.secondary_loss_type == 'lovasz':
            # lovasz_softmax expects logits before softmax
            # Ensure that lovasz_softmax is correctly implemented or imported
            self.secondary = lovasz_softmax
        else:
            raise ValueError(f"Unsupported secondary loss type: {secondary_loss}. Choose 'miou' or 'lovasz'.")

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        
        if self.secondary_loss_type == 'miou':
            miou_loss = self.secondary(logits, target)
            return ce_loss + self.weight_secondary * miou_loss
        elif self.secondary_loss_type == 'lovasz':
            lovasz_loss = self.secondary(F.softmax(logits, dim=1), target)
            return ce_loss + self.weight_secondary * lovasz_loss