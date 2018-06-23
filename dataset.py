import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

CLIP_MAX = 0.5
CLIP_MIN = -0.5

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        return x, y, idx

    def __len__(self):
        return len(self.X)

def get_dataset():
    data_tr = datasets.MNIST('./MNIST', train=True, download=True)
    data_te = datasets.MNIST('./MNIST', train=False, download=True)
    X_tr, Y_tr = data_tr.train_data, data_tr.train_labels
    X_te, Y_te = data_te.test_data, data_te.test_labels
    X_tr = X_tr.float()/255.0 - (1.0 - CLIP_MAX)
    X_te = X_te.float()/255.0 - (1.0 - CLIP_MAX)
    X_tr.unsqueeze_(1)
    X_te.unsqueeze_(1)
        
    return X_tr, Y_tr, X_te, Y_te

def recover_image(x):
    img = (x + 0.5)*255
    img = Image.fromarray(img).convert('RGB')
    return img

def recover_noise(x):
    img = np.ones((x.shape[0], x.shape[1], 3))
    img[x<0, 1] += x[x<0]
    img[x<0, 2] += x[x<0]
    img[x>0, 0] -= x[x>0]
    img[x>0, 2] -= x[x>0]
    img *= 255
    img = img.astype('uint8')
    img = Image.fromarray(img.astype('uint8'))
    return img
