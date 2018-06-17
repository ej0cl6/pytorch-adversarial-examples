import torch
import torch.nn.functional as F

class FGSM:
    """
    Fast Gradient Sign Method
    Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy.
    Explaining and Harnessing Adversarial Examples.
    ICLR, 2015
    """

    def __init__(self, eps=0.15, clip_max=0.5, clip_min=-0.5):
        self.eps = eps
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        ny = torch.unsqueeze(y, 0)
        nx.requires_grad_()
        out = model(nx)
        loss = F.cross_entropy(out, ny)
        loss.backward()
        x_adv = nx + self.eps * torch.sign(nx.grad.data)
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()

class BIM:
    """
    Basic Iterative Method
    Alexey Kurakin, Ian J. Goodfellow, Samy Bengio.
    Adversarial Examples in the Physical World.
    arXiv, 2016
    """

    def __init__(self, eps=0.15, eps_iter=0.01, n_iter=50, clip_max=0.5, clip_min=-0.5):
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_iter = n_iter
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        ny = torch.unsqueeze(y, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        for i in range(self.n_iter):
            out = model(nx+eta)
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps_iter * torch.sign(nx.grad.data)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()

        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()
