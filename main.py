import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MyDataset, get_dataset, recover_image, recover_noise
from model import Net, train, predict
from attackers import FGSM, BIM, DeepFool

SEED = 1

CLIP_MAX = 0.5
CLIP_MIN = -0.5

args = {'n_epoch': 10,
        'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
        'loader_te_args': {'batch_size': 10, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5}}

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

X_tr, Y_tr, X_te, Y_te = get_dataset()
loader_tr = DataLoader(MyDataset(X_tr, Y_tr), shuffle=True, **args['loader_tr_args'])

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), **args['optimizer_args'])
train(model, device, loader_tr, optimizer, args['n_epoch'])

attacker = FGSM(eps=0.15, clip_max=CLIP_MAX, clip_min=CLIP_MIN)
# attacker = BIM(eps=0.15, eps_iter=0.01, n_iter=50, clip_max=CLIP_MAX, clip_min=CLIP_MIN)
# attacker = DeepFool(max_iter=50, clip_max=CLIP_MAX, clip_min=CLIP_MIN)
print('attacker: {}'.format(type(attacker).__name__))

demo_idxs = [545, 107, 38, 142, 65, 15, 21, 171, 257, 20]
X_te_cln = X_te[demo_idxs]
Y_te_cln = Y_te[demo_idxs]
X_te_adv = torch.zeros(X_te_cln.shape)
model.cpu()
for i in range(len(X_te_cln)):
    X_te_adv[i] = attacker.generate(model, X_te_cln[i], Y_te_cln[i])

loader_te_cln = DataLoader(MyDataset(X_te_cln, Y_te_cln), shuffle=False, **args['loader_te_args'])
loader_te_adv = DataLoader(MyDataset(X_te_adv, Y_te_cln), shuffle=False, **args['loader_te_args'])

model.cuda()
P_cln = predict(model, device, loader_te_cln)
P_adv = predict(model, device, loader_te_adv)

print('labels of clean images:       {}'.format(P_cln.numpy()))
print('labels of adversarial images: {}'.format(P_adv.numpy()))

for i in range(10):
    recover_image(X_te_cln.numpy()[i][0]).save('results/Clean/{}.png'.format(i))
    recover_image(X_te_adv.numpy()[i][0]).save('results/{}/{}_adversarial.png'.format(type(attacker).__name__, i))
    recover_noise(X_te_adv.numpy()[i][0]-X_te_cln.numpy()[i][0]).save('results/{}/{}_diff.png'.format(type(attacker).__name__, i))
