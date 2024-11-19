import os
import re

import cv2
import time
import numpy as np
from segmentation_models_pytorch.losses import DiceLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from networks.net_factory import net_factory
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Resize, RandomCrop, VerticalFlip, HorizontalFlip, Normalize, Compose)
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


def get_transforms(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(),
                VerticalFlip()
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


class JetDataset(Dataset):
    def __init__(self, idx, image_path, phase="train"):
        assert phase in ("train", "val", "test")
        self.idx = idx
        self.image_path = image_path
        self.phase = phase
        self.transform = get_transforms(phase)

    def __getitem__(self, index):
        global extracted_number
        real_idx = self.idx[index]

        image_list = sorted(
            [os.path.join(f'{self.image_path}/image', i) for i in os.listdir(f'{self.image_path}/image')])
        label_list = sorted(
            [os.path.join(f'{self.image_path}/label', i) for i in os.listdir(f'{self.image_path}/label')])

        image_path = image_list[real_idx]
        mask_path = label_list[real_idx]

        match = re.search(r'(\d+)', image_path)

        if match:
            extracted_number = match.group(1)

        image = cv2.imread(image_path)
        image = image[:, :, ::-1].copy()
        mask = cv2.imread(mask_path)

        image = self.transform(image=image)
        label = self.transform(image=mask / 255)

        return image['image'].float(), torch.unsqueeze(label['image'][2, ...], 0).float(), extracted_number

    def __len__(self):
        return len(self.idx)


def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def metric(probability, truth, threshold=0.5):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.contiguous().view(batch_size, -1)
        truth = truth.contiguous().view(batch_size, -1)
        assert (probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


class Meter:

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]

        iou = np.nanmean(self.iou_scores)
        return dices, iou


def epoch_log(phase, epoch, epoch_loss, meter, start):
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.6f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (
        epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    ious = []

    preds = np.copy(outputs)
    labels = np.array(labels)

    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def provider(
        path,
        phase,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=0,
):
    file_list = sorted([os.path.join(f'{path}/image', i) for i in os.listdir(f'{path}/image')])
    train_temp_idx, test_idx = train_test_split(range(len(file_list)), random_state=66666, test_size=0.1)

    train_idx, val_idx = train_test_split(range(len(train_temp_idx)), random_state=66666, test_size=0.1)

    if phase == 'train':
        index = train_idx
    elif phase == 'validation':
        index = val_idx
    else:
        index = test_idx

    dataset = JetDataset(index, path, phase=phase)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        generator=torch.Generator(device='cuda')
    )

    return dataloader


class Trainer(object):

    def __init__(self, model):
        self.num_workers = 0
        self.batch_size = {"train": 8, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 1e-4
        self.num_epochs = 500

        self.best_loss = float("inf")
        self.best_dice = float(0)
        self.phases = ["train", "val"]
        self.device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=4, verbose=True)
        self.net = self.net.to(self.device)
        # https://blog.csdn.net/AugustMe/article/details/108364073
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                path=path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)

        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | : {start}")
        # https://blog.csdn.net/weixin_44211968/article/details/123774649

        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets, _ = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps

            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            else:
                pass

            running_loss += loss.item()
            outputs = outputs.detach().cpu()

            meter.update(targets, outputs)

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()

        return epoch_loss, dice

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss, dice = self.iterate(epoch, "val")

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                self.best_dice = dice
                torch.save(state, "./model.pth")

            print()


MODELPATH = "./model.pth"
path = './jet_stream_dataset'

if os.path.exists(MODELPATH):

    model = net_factory(net_type="raunet++", in_chns=3, class_num=3)
    state = torch.load(MODELPATH, map_location=lambda storage, loc: storage)

    model.load_state_dict(state["state_dict"])
else:
    model = smp.UnetPlusPlus('resnet18', classes=1, activation=None)

model_trainer = Trainer(model)

model_trainer.start()
