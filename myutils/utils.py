import copy
import os
import random
import torch.nn.functional as F
import numpy as np
import torch
import torch.backends.cudnn as cudnn


# --------------------------------------------------------------------------------
# Define seed
# --------------------------------------------------------------------------------
def fix_seed_for_reproducibility(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# --------------------------------------------------------------------------------
# Define EMA: Mean Teacher Framework
# --------------------------------------------------------------------------------
class EMA(object):
    def __init__(self, model, total_step):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.total_step = total_step
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        decay = np.exp2(self.step/self.total_step)-1
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

# --------------------------------------------------------------------------------
# Define Evaluation Metrics
# --------------------------------------------------------------------------------
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def empty_mat(self):
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >=0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self, bk=False):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        precison = torch.diag(h) / torch.sum(h, dim=0)
        recall = torch.diag(h) / torch.sum(h, dim=1)
        fscore = 2 * precison * recall / (precison + recall)
        pe = torch.sum(torch.sum(h, dim=0) * torch.sum(h, dim=1))/ (h.sum()*h.sum())
        kappa = (acc-pe)/(1-pe)

        if bk:
            return torch.mean(iu).item(), torch.mean(fscore).item(), acc.item(), kappa.item()
        else:
            return torch.mean(iu[1:]).item(), torch.mean(fscore[1:]).item(), acc.item(), kappa.item()


# --------------------------------------------------------------------------------
# Define Polynomial Decay
# --------------------------------------------------------------------------------
from torch.optim.lr_scheduler import _LRScheduler
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

# --------------------------------------------------------------------------------
# Define Training Losses
# --------------------------------------------------------------------------------

def compute_supervised_loss(predict, target):

    loss = F.cross_entropy(predict, target, reduction='mean')
    return loss


def compute_unsupervised_loss(predict, target, logits, threshold=0.7, mode='stratified_sampling'):
    class_num = predict.shape[1]

    if mode == 'ce_loss':
        loss = F.cross_entropy(predict, target, reduction='none')
        mask = logits > threshold
        loss = torch.mean(torch.masked_select(loss, mask))

    elif mode == 'class_threshold':
        with torch.no_grad():

            numlist = torch.bincount(target.flatten(), minlength=class_num)
            numlist = numlist / numlist.max()
            # print(threshold * (0.1 * numlist + 0.9))

            mask = logits.ge(threshold * (0.1 * numlist[target] + 0.9)).float()  # Linear

        loss = F.cross_entropy(predict, target, reduction='none')
        loss = torch.mean(torch.masked_select(loss, mask.bool()))

    return loss, mask

def compute_caco_loss(rep, label, mask, prob, strong_threshold=1.0, temp=0.5):

    num_segments = label.shape[1]
    device = rep.device
    # compute valid binary mask for each pixel
    valid_pixel = label * mask
    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = prob[:, i, :, :]
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries
        seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))  # class-mean features
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))  # valid samples per class（shape: num_class）

    # compute contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        caco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)

        for i in range(valid_seg):
            if len(seg_feat_hard_list[i]) > 0:
                anchor_feat_hard = seg_feat_hard_list[i]
            else:
                continue

            features = F.normalize(anchor_feat_hard, dim=1)  # shape: pixel_num * dim
            y_center = F.normalize(seg_proto, dim=1)  # shape: class_num * dim

            similarity_matrix = torch.matmul(features, y_center.T)  # shape: pixel_num * class_num
            caco_loss = caco_loss + F.cross_entropy(similarity_matrix / temp, (torch.ones(len(anchor_feat_hard))*i).long().to(device))
        return caco_loss / valid_seg

def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)

    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)

