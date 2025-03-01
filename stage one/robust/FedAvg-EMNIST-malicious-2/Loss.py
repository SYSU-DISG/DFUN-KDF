import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class KL_Loss(nn.Module):
    def __init__(self, temperature=3):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        loss =nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # loss = -torch.sum(torch.sum(torch.mul(teacher_outputs, output_batch)))/teacher_outputs.size(0)
        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss
def Loss(outputs,global_logits,labels):
    criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
    criterion_KL = KL_Loss(temperature=1)
    loss_true = criterion(outputs, labels)
    loss_kd = criterion_KL.forward(outputs, global_logits)
    loss = loss_true + 0.4 * loss_kd
    return loss

def Loss_mse(outputs,global_logits):
    criterion = nn.MSELoss()
    output_batch = F.log_softmax(outputs)
    teacher_outputs = F.softmax(outputs) + 10 ** (-7)
    loss = criterion(output_batch,teacher_outputs)
    return loss