import time
import numpy as np
import os
import scipy.io
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sample_data')
parser.add_argument('--method', type=str, default='LCFI')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--lambda1', type=float, default=0.1)
parser.add_argument('--lambda2', type=float, default=0.1)
parser.add_argument('--lambda_struct', type=float, default=0.1)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--valid_size', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda') 
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

from utils.metrics import evaluation_KLD, evaluation_lt
from utils.utils import set_seed

from methods.LCFI import LCFI

set_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device 

print(f"Using device: {args.device}")

mat_file_path = os.path.join(r'C:\Users\Administrator\Desktop\LCFI\dataset', f'{args.dataset}.mat')
mat_data = scipy.io.loadmat(mat_file_path)
X = mat_data['features']   
Y = mat_data['labels']
Y = Y.astype(np.float32)
if Y.ndim == 1:
    Y = Y[:, None]

def get_model_by_name(method, x_train, y_train, args, label_corr):
    if method == 'LCFI':
        return LCFI(num_features=x_train.shape[1], num_labels=y_train.shape[1],
                   hidden_dim=args.hidden_dim, lr=args.lr, weight_decay=1e-4, device=args.device)
    else:
        raise ValueError(f"Unknown method: {method}")

def train_fold(model, train_loader, val_loader, max_epoch, train_path):
    best_state_dict = None
    best_optimizer_state_dict = None
    best_epoch = 0
    min_result = np.inf

    for epoch in range(max_epoch):
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            mask_batch = torch.ones_like(y_batch, device=model.device)

            if hasattr(model, "train_step"):
                train_loss = model.train_step(x_batch, y_batch, mask_batch)
            else:
                train_loss = model.fit(x_batch.cpu().numpy(), y_batch.cpu().numpy())
            total_train_loss += train_loss if train_loss is not None else 0

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{max_epoch}, Training Loss: {avg_train_loss:.4f}')

        preds, ys = model.get_result(val_loader)
        val_kld = evaluation_KLD(ys, preds)

        if val_kld < min_result:
            min_result = val_kld
            best_state_dict = copy.deepcopy(getattr(model, "state_dict", lambda: None)())
            best_optimizer_state_dict = copy.deepcopy(getattr(getattr(model, "optimizer", None), "state_dict", lambda: None)())
            best_epoch = epoch
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_state_dict,
                'optimizer_state_dict': best_optimizer_state_dict,
            }, os.path.join(train_path, 'best.tar'))

    if hasattr(model, "save"):
        model.save(os.path.join(train_path, "last.pt"), epoch=max_epoch - 1)
    return min_result

def test_fold(model, train_path, test_loader):
    if hasattr(model, "load"):
        model.load(os.path.join(train_path, "best.tar"))
    preds, ys = model.get_result(test_loader)
    return preds


kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
metric_names = ['Chebyshev', 'Clark', 'Canberra', 'KLD', 'Cosine', 'Intersection']
metric_results = {name: [] for name in metric_names}
folds_time = []

label_correlation_matrix = np.ones((Y.shape[1], Y.shape[1]))


fold = 0  
train_idx, test_idx = list(kf.split(X))[fold]

x_train_all, y_train_all = X[train_idx], Y[train_idx]
x_test, y_test = X[test_idx], Y[test_idx]

inner_kf = KFold(n_splits=9, shuffle=True, random_state=args.seed)
ti, vi = list(inner_kf.split(x_train_all))[0]
x_train, y_train = x_train_all[ti], y_train_all[ti]
x_val, y_val = x_train_all[vi], y_train_all[vi]

train_dataset = TensorDataset(torch.from_numpy(x_train).float(),
                              torch.from_numpy(y_train).float())
val_dataset = TensorDataset(torch.from_numpy(x_val).float(),
                            torch.from_numpy(y_val).float())
test_dataset = TensorDataset(torch.from_numpy(x_test).float(),
                             torch.from_numpy(y_test).float())

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_path = os.path.join('save', 'lt', f'{args.method}', f'fold_{fold}', f'{args.dataset}')
os.makedirs(train_path, exist_ok=True)

model = get_model_by_name(args.method, x_train, y_train, args, label_correlation_matrix)
model.to(args.device)  

t0 = time.time()
train_fold(model, train_loader, val_loader, args.max_epoch, train_path)
y_pred = test_fold(model, train_path, test_loader)
folds_time.append(time.time() - t0)

result = evaluation_lt(y_test, y_pred, y_train=y_train_all)
for name in metric_names:
    metric_results[name].append(result[name])
print('Fold %d results:' % (fold+1),
      ', '.join(['%s: %.4f' % (n, result[n]) for n in metric_names]))

print('\n=== Final Results ===')
for name in metric_names:
    arr = np.array(metric_results[name])
    mean = arr.mean()
    max_minus_mean = np.abs(arr.max() - mean)
    vals = ', '.join([f'{v:.4f}' for v in arr])
    print(f'{name}: {vals}\n Mean: {mean:.4f}')
