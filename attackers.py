import torch
import torch.nn.functional as F

class FGSM:
    def __init__(self, eps=0.15, clip_max=0.5, clip_min=-0.5):
        self.eps = eps
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        ny = torch.unsqueeze(y, 0)
        nx.requires_grad = True
        out = model(nx)
        loss = F.cross_entropy(out, ny)
        loss.backward()
        x_adv = nx + self.eps * torch.sign(nx.grad.data)
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()
