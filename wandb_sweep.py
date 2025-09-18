import datetime
import os
import random
import pytz

import PIL
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import ConcatDataset, DataLoader

import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from network import Resnet
from utils import Cutout, CrossEntropyLabelSmooth, cal_mean_std, EarlyStopException


def seed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    random.seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def train(cfg, run, mean, std, model, train_loader, dev_loader):
    # criterion = nn.CrossEntropyLoss()
    criterion = CrossEntropyLabelSmooth().cuda()

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'], eta_min=cfg['lr'] / 100)

    stale = 0
    best_acc = -torch.inf
    for epoch in range(cfg['n_epochs']):
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader)
        pbar.set_description(f"T : {epoch + 1:03d}/{cfg['n_epochs']:03d}")
        for images, labels in pbar:
            # # 检查是否被 Sweep 或手动停止
            # if hasattr(run, 'stopped') and run.stopped:
            #     print(f"HyperBand Run {run.id} stopped by at epoch {epoch + 1}")
            #     run.log({"early_stop": True, "stop_epoch": epoch + 1})
            #     run.finish(exit_code=0)
            #     raise EarlyStopException

            images = images.cuda()
            labels = labels.cuda()

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['grad_norm_max'])

            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            pbar.update()
            pbar.set_postfix({'lr': lr, 'batch_loss': loss.item(), 'batch_acc': acc.item(),
                              'loss': sum(train_loss) / len(train_loss),
                              'acc': sum(train_accs).item() / len(train_accs)})

        scheduler.step()

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        pbar.close()

        cn_tz = pytz.timezone('Asia/Shanghai')
        now_time = datetime.datetime.now(cn_tz).strftime('%Y-%m-%d %H:%M:%S')

        # Print the information.
        with open(f"{cfg['save_path']}/{cfg['exp_name']}_log.txt", "a"):
            print(f"@{now_time}, [ Train | {epoch + 1:03d}/{cfg['n_epochs']:03d} ] loss = {train_loss:.5f}, "
                  f"lr = {optimizer.param_groups[0]['lr']:.5f}, acc = {train_acc:.5f}")

        run.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "now_lr": lr})

        # ---------- Validation ----------
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        good_cases = wandb.Table(columns=['valid_right_Image', 'GroundTruth', 'Prediction'])
        bad_cases = wandb.Table(columns=['valid_error_Image', 'GroundTruth', 'Prediction'])

        # Iterate the validation set by batches.
        pbar = tqdm(dev_loader)
        pbar.set_description(f"Valid: {epoch + 1:03d}/{cfg['n_epochs']:03d}")
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()

            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(images)

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels)

            # Compute the accuracy for current batch.
            y_pred = logits.argmax(dim=-1)
            acc = (y_pred == labels).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            pbar.update()
            pbar.set_postfix({'valid_loss': sum(valid_loss) / len(valid_loss),
                              'valid_acc': sum(valid_accs).item() / len(valid_accs)})

            mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
            std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
            img = images.cpu().permute(0, 2, 3, 1).numpy()
            img = np.clip(img * std + mean, 0, 1)
            for i in range(images.size(0)):
                # log badcase
                if y_pred[i] != labels[i] and len(bad_cases.data) < 100:
                    bad_cases.add_data(
                        wandb.Image(PIL.Image.fromarray((img[i] * 255).astype(np.uint8))),
                        labels[i].item(),
                        y_pred[i].item()
                    )
                # log goodcase
                elif y_pred[i] == labels[i] and len(good_cases.data) < 10:
                    good_cases.add_data(
                        wandb.Image(PIL.Image.fromarray((img[i] * 255).astype(np.uint8))),
                        labels[i].item(),
                        y_pred[i].item()
                    )

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        pbar.close()

        cn_tz = pytz.timezone('Asia/Shanghai')
        now_time = datetime.datetime.now(cn_tz).strftime('%Y-%m-%d %H:%M:%S')

        # update logs
        if valid_acc > best_acc:
            with open(f"{cfg['save_path']}/{cfg['exp_name']}_log.txt", "a"):
                print(
                    f"@{now_time}, [Valid | {epoch + 1:03d}/{cfg['n_epochs']:03d} ] "
                    f"loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"{cfg['save_path']}/{cfg['exp_name']}_log.txt", "a"):
                print(
                    f"@{now_time}, [Valid | {epoch + 1:03d}/{cfg['n_epochs']:03d} ] "
                    f"loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        run.log({"epoch": epoch + 1, "valid_loss": valid_loss, "valid_acc": valid_acc,
                 'test_good_cases': good_cases, 'test_bad_cases': bad_cases})

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, acc={valid_acc:.5f}, saving model")
            torch.save(model.state_dict(), f"{cfg['save_path']}/{cfg['exp_name']}_best.ckpt")
            # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > cfg['patience']:
                print(f"No improvment {cfg['patience']} consecutive epochs, early stopping")
                break

        if hasattr(run, 'stopped') and run.stopped:
            print(f"HyperBand Run {run.id} early stopped at epoch {epoch + 1}")
            run.log({"early_stop": True, "stop_epoch": epoch + 1})
            run.finish()
            raise EarlyStopException


def test(cfg, run, mean, std, model, test_loader):
    good_cases = wandb.Table(columns=['test_right_Image', 'GroundTruth', 'Prediction'])
    bad_cases = wandb.Table(columns=['test_error_Image', 'GroundTruth', 'Prediction'])

    model.load_state_dict(torch.load(f"{cfg['save_path']}/{cfg['exp_name']}_best.ckpt", map_location='cpu'))

    # Start evaluate
    model.eval()

    test_accs = []
    # Iterate the validation set by batches.
    pbar = tqdm(test_loader)
    for images, lables in pbar:
        images = images.cuda()
        labels = lables.cuda()
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(images)
            # Compute the accuracy for current batch.
            y_pred = logits.argmax(dim=-1)
            acc = (y_pred == labels).float().mean()
            test_accs.append(acc)
        pbar.update()
        pbar.set_postfix({'test_acc': sum(test_accs).item() / len(test_accs)})

        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
        img = images.cpu().permute(0, 2, 3, 1).numpy()
        img = np.clip(img * std + mean, 0, 1)
        for i in range(images.size(0)):
            # log badcase
            if y_pred[i] != labels[i] and len(bad_cases.data) < 100:
                bad_cases.add_data(
                    wandb.Image(PIL.Image.fromarray((img[i] * 255).astype(np.uint8))),
                    labels[i].item(),
                    y_pred[i].item()
                )
            # log goodcase
            elif y_pred[i] == labels[i] and len(good_cases.data) < 10:
                good_cases.add_data(
                    wandb.Image(PIL.Image.fromarray((img[i] * 255).astype(np.uint8))),
                    labels[i].item(),
                    y_pred[i].item()
                )

    cn_tz = pytz.timezone('Asia/Shanghai')
    now_time = datetime.datetime.now(cn_tz).strftime('%Y-%m-%d %H:%M:%S')

    test_acc = sum(test_accs) / len(test_accs)
    with open(f"{cfg['save_path']}/{cfg['exp_name']}_log.txt", "a"):
        print(f"@{now_time}, test_acc = {test_acc:.5f}")

    run.log({'test_good_cases': good_cases, 'test_bad_cases': bad_cases, "test_acc": test_acc})


def log_artifacts(run, cfg):
    arti_dataset = wandb.Artifact('Cifar10', type='dataset')
    arti_dataset.add_dir('./CIFAR10')
    run.log_artifact(arti_dataset)

    arti_code = wandb.Artifact('py', type='code')
    arti_code.add_file('./main.py')
    arti_code.add_file('./network.py')
    run.log_artifact(arti_code)

    arti_model = wandb.Artifact('Resnet18', type='model')
    arti_model.add_file(f"{cfg['save_path']}/{cfg['exp_name']}_best.ckpt")
    run.log_artifact(arti_model)

    arti_outputs = wandb.Artifact('outputs', type='outputs')
    arti_outputs.add_dir('./outputs')
    run.log_artifact(arti_outputs)


# 用一个中间函数将训练和测试包起来，并传入参数
def train_test():
    try:
        cn_tz = pytz.timezone('Asia/Shanghai')
        now_time = datetime.datetime.now(cn_tz).strftime('%Y-%m-%d %H:%M:%S')

        run = wandb.init(name=now_time, save_code=True)  # Sweep 会注入超参
        cfg = dict(run.config)  # 拿到动态配置

        # sweep每次启动子进程，要在内部设置种子才能保证初始环境一致（种子可作为超参数）
        seed(cfg['seed'])

        mean, std = cal_mean_std(cfg)

        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop((32, 32), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(0.5),  # 增加水平翻转
            transforms.RandomAffine(15),
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            Cutout(n_holes=cfg['cutout_holes'], length=cfg['cutout_len'])
        ])

        # 从训练集划分验证集
        full_train = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=None)  # 不指定 transform
        train_idx, dev_idx = train_test_split(
            range(len(full_train)), test_size=5000, random_state=42, stratify=full_train.targets)

        # 训练集，验证集，测试集
        train_set = datasets.CIFAR10(root='./CIFAR10', train=True, transform=train_tfm)
        dev_set = datasets.CIFAR10(root='./CIFAR10', train=True, transform=test_tfm)
        test_set = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=test_tfm)

        train_set = torch.utils.data.Subset(train_set, train_idx)
        dev_set = torch.utils.data.Subset(dev_set, dev_idx)

        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        dev_loader = DataLoader(dev_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

        model = Resnet(dropout=cfg['dropout']).cuda()

        train(cfg, run, mean, std, model, train_loader, dev_loader)
        test(cfg, run, mean, std, model, test_loader)

        # artifact
        log_artifacts(run, cfg)
        run.finish()
    except EarlyStopException:
        pass


def get_available_gpus():
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    else:
        return []


def run_agent_on_gpu(gpu_id, sweep_id, count):
    # 设置可见 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Agent starting on GPU {gpu_id}, sweep_id={sweep_id}")

    # 启动 agent（不指定 count，持续领取任务直到 sweep 结束）
    wandb.agent(sweep_id, function=train_test, count=count)


if __name__ == "__main__":
    cfg = {
        'dataset_root': './CIFAR10',
        'save_path': './outputs',
        'exp_name': "cifar_10",
        'batch_size': 64,
        'lr': 3e-3,
        'seed': 20250903,
        'weight_decay': 1e-5,
        'grad_norm_max': 10,
        'n_epochs': 300,
        'patience': 30,
    }

    # os.environ.pop('WANDB_API_KEY', None)  # 清理环境变量
    wandb.login()

    os.makedirs(cfg['save_path'], exist_ok=True)

    # wandb自动调优参数配置
    sweep_config = {'method': 'random'}
    # 优化指标
    metrics = {'name': 'valid_loss',
               'goal': 'minimize'}
    sweep_config['metric'] = metrics

    sweep_config['parameters'] = {}
    # 设置参数
    sweep_config['parameters'].update({'dataset_root': {'value': cfg['dataset_root']},
                                       'save_path': {'value': cfg['save_path']},
                                       'exp_name': {'value': cfg['exp_name']},
                                       'n_epochs': {'value': cfg['n_epochs']},
                                       'seed': {'value': cfg['seed']},
                                       'patience': {'value': cfg['patience']},
                                       'cutout_len': {'value': 16},
                                       'cutout_holes': {'value': 1},
                                       # 离散参数
                                       'batch_size': {'values': [32, 64, 128, 256, 512]},
                                       # 连续参数
                                       'dropout': {'distribution': 'q_uniform', 'q': 0.1, 'max': 0.6, 'min': 0.2},
                                       'lr': {'distribution': 'log_uniform_values', 'max': 0.1, 'min': 1e-6},
                                       'weight_decay': {'distribution': 'log_uniform_values', 'max': 1e-4, 'min': 1e-5},
                                       'grad_norm_max': {'distribution': 'q_uniform', 'q': 2, 'max': 16, 'min': 4}
                                       })

    # 早停没有希望的任务
    sweep_config['early_terminate'] = {'type': 'hyperband', 'min_iter': 10, 'eta': 3}

    # # 在可用GPU上启动agent
    # available_gpus = get_available_gpus()
    # print(f"Available GPUs: {available_gpus}")
    #
    # processes = []
    # mp.set_start_method('spawn')  # ← 强制使用 spawn，避免 fork 导致CUDA 上下文初始化冲突
    # for gpu_id in available_gpus:
    #     p = mp.Process(target=run_agent_on_gpu, args=(gpu_id, sweep_id, 27))
    #     p.start()
    #     processes.append(p)
    #
    # # 所有进程
    # for p in processes:
    #     p.join()

    sweep_id = wandb.sweep(sweep_config, project=cfg['exp_name'])
    wandb.agent(sweep_id, function=train_test, count=27)
