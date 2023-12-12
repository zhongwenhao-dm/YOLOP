import numpy as np
import torch
from pdb import set_trace as ps 
import scipy.spatial as spt


def area_aware_loss_for_ll(ll_pre, da_gt):
    score = 0
    # ll_pre,da_gt : torch.Size([48, 2, 384, 640])
    # 从da_gt中值为1的点中，找到ll_pre中最近的点的概率p，然后累加1-p

    da_idx = np.array(np.where(da_gt==0)).T
    ll_idx = np.array(np.where(ll_pre < 0.9)).T

    tree = spt.cKDTree(data=ll_idx)

    for i in range(len(da_idx)):
        point = da_idx[i]
        distance, index = tree.query(point, k=1)
        # print('距离{}最近的是{}，距离为{:.2f}，概率为{:.3f}'.format(point, ll_idx[index], distance, ll_pre[ll_idx[index][0],ll_idx[index][1]]))
        score += 1-ll_pre[ll_idx[index][0],ll_idx[index][1]]

    return score/(384*640)

def boundary_aware_loss_for_da(da_pre, ll_gt):
    score = 0


    return score


# 文件相关路径
root = '/home/user/ZWH/code/YOLOP-main/'
ll_pre_file = root + 'll_pre00.txt'
da_gt_file = root + 'da_gt00.txt'
ll_pre01 = root + 'll_pre01.txt'
da_gt01 = root + 'da_gt01.txt'

da_pre_file = root + 'da_pre00.txt'
ll_gt_file = root + 'll_gt00.txt'



# 读入文件
# da_gt00中，0就是划分出来的da；da_gt01则相反
# ll_pre00中，小于1的就是划分的ll
ll_pre = np.loadtxt(ll_pre_file)
da_gt = np.loadtxt(da_gt_file)
ll_pre01 = np.loadtxt(ll_pre01)
da_gt01 = np.loadtxt(da_gt01)

da_pre = np.loadtxt(da_pre_file)
ll_gt = np.loadtxt(ll_gt_file)


td = torch.Tensor(da_gt)
tl = torch.Tensor(ll_pre)
td01 = torch.Tensor(da_gt01)
tl01 = torch.Tensor(ll_pre01)

laa = area_aware_loss_for_ll(ll_pre, da_gt)
print(laa)
lba = boundary_aware_loss_for_da(da_pre, ll_gt)
print(lba)



