#main code for computing the loss functions
import torch
import torch.nn as nn

class GramMatrix_gen(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) #batch matrix multiplication with the transposed version
        G.div_(h*w) #normalization constant
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix_gen()(input), target)
        return(out)


