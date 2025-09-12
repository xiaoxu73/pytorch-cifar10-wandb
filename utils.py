import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset, DataLoader, Subset


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=10, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


# Cutout， 在输入图片上添加方形掩码
class Cutout:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class EarlyStopException(Exception):
    pass


def cal_mean_std(cfg):
    # 用临时 transform 计算均值/方差
    temp_tfm = transforms.Compose([
        transforms.ToTensor()
    ])

    # 下载数据集
    temp_train_set = datasets.CIFAR10(root=cfg['dataset_root'], train=True, download=True, transform=temp_tfm)
    temp_test_set = datasets.CIFAR10(root=cfg['dataset_root'], train=False, download=True, transform=temp_tfm)

    # Construct datasets.
    temp_dataset = ConcatDataset([
        temp_train_set,
        temp_test_set
    ])

    temp_loader = DataLoader(temp_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    for images, _ in temp_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std:  {std}")

    return mean, std


def data2fig(data):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(data)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def fig2img(fig):
    import io, PIL
    import matplotlib.pyplot as plt
    """Convert a Matplotlib figure to PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = PIL.Image.open(buf)
    # 让 PIL 对象脱离 buffer 独立存在
    img = img.copy()
    buf.close()
    plt.close(fig)  # 防止内存泄漏
    return img
