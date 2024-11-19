import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders import utils
from dataloaders.dataset import (TwoStreamBatchSampler, MyDataSets)
from networks.net_factory import net_factory
from utils_ import losses, metrics, ramps
from val_2D import test_single_volume_my
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='../data/jet_stream_dataset_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='jet_stream/Cross_Pseudo_Supervision', help='experiment_name')

parser.add_argument('--model', type=str,
                    default='unet++', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=100000, help='maximum epoch number to train')

parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=4,
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=66666, help='random seed')

parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')

parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=5,
                    help='labeled data')

parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=150.0, help='consistency_rampup')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_iou_dice(output, label):
    output_img = (output > 0).astype(np.uint8)
    label_img = (label > 0).astype(np.uint8)

    intersection = np.logical_and(output_img, label_img)

    union = np.logical_or(output_img, label_img)

    iou = np.sum(intersection) / np.sum(union)

    dice_coefficient = 2 * np.sum(intersection) / (np.sum(output_img) + np.sum(label_img))

    return iou, dice_coefficient


def filter_pseudo_labels(images, model_path, threshold) -> bool:
    def add_gaussian_noise_3d(tensor, noise_std=4):

        noise = torch.normal(mean=0.0, std=noise_std, size=tensor.size())
        return tensor + noise

    def apply_feature_masking_3d(tensor, mask_ratio=0.3):

        n_features = tensor.size(2)
        n_masked_features = int(n_features * mask_ratio)

        masked_tensor = tensor.clone()
        for i in range(tensor.size(0)):
            for j in range(tensor.size(1)):
                mask_indices = torch.randperm(n_features)[:n_masked_features]

                masked_tensor[i, j, mask_indices] = 0
        return masked_tensor

    def add_laplace_noise_3d(tensor, noise_scale=1.0):

        laplace_dist = torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale)  # 位置参数设为 0，尺度参数为 noise_scale
        noise = laplace_dist.sample(tensor.size())
        return tensor + noise

    MODELPATH = model_path

    model = net_factory(net_type=args.model, in_chns=3, class_num=3)

    state = torch.load(MODELPATH)
    model.load_state_dict(state)

    feature = model.encoder(images)

    feature[5] = add_gaussian_noise_3d(feature[5])

    output1 = model.decoder(*feature)
    output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")

    output = model(images).squeeze(0).squeeze(0).detach().numpy()
    output = np.where(output > 0.5, 255, 0).astype("uint8")

    iou1, _ = calculate_iou_dice(output1, output)

    feature[5] = add_laplace_noise_3d(feature[5])

    output1 = model.decoder(*feature)
    output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")

    output = model(images).squeeze(0).squeeze(0).detach().numpy()
    output = np.where(output > 0.5, 255, 0).astype("uint8")

    iou2, _ = calculate_iou_dice(output1, output)

    feature[5] = apply_feature_masking_3d(feature[5])

    output1 = model.decoder(*feature)
    output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")

    output = model(images).squeeze(0).squeeze(0).detach().numpy()
    output = np.where(output > 0.5, 255, 0).astype("uint8")

    iou3, _ = calculate_iou_dice(output1, output)

    return (iou1 + iou2 + iou3) / 3.0 > threshold


weak_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

strong_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
])


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "jet_stream" in dataset:
        ref_dict = {"5": 40, "10": 80, "20": 161, "30": 242}

    return ref_dict[str(patiens_num)]


class DataSetsWithPerturbation(MyDataSets):
    def __init__(self, base_dir, split, num=None, labeled_idxs=None):
        super().__init__(base_dir, split, num)
        self.labeled_idxs = labeled_idxs

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        image_np = sample['image']

        if image_np.shape == (3, 320, 512):  # (C, H, W)-->(H, W, C)
            image_np = np.transpose(image_np, (1, 2, 0))
        elif image_np.shape == (1, 1, 512):
            raise ValueError(f"Unexpected shape at index {idx}: {image_np.shape}")

        if image_np.shape != (320, 512, 3):
            print(f"Warning: Adjusted shape at index {idx} is still incorrect: {image_np.shape}")

        image_pil = Image.fromarray(image_np)

        if idx in self.labeled_idxs:
            sample['image'] = weak_augmentation(image_pil)
        else:
            sample['image'] = strong_augmentation(image_pil)

        if not isinstance(sample['image'], torch.Tensor):
            sample['image'] = transforms.ToTensor()(sample['image'])

        return sample


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):

        model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = DataSetsWithPerturbation(base_dir=args.root_path, split="train", num=None,
                                        labeled_idxs=list(
                                            range(0, patients_to_slices(args.root_path, args.labeled_num))))
    db_val = MyDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, date = sampled_batch['image'], sampled_batch['label'], sampled_batch['date']
            volume_batch, label_batch = volume_batch.cuda().float(), label_batch.cuda().float()
            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss1 = 0.5 * ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss2 = 0.5 * ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            # save pseudo_outputs
            if epoch_num + 1 > 50:
                if filter_pseudo_labels(volume_batch[1], '../model/model.pth', 0.8):
                    torchvision.utils.save_image(pseudo_outputs1, f'{date[1]}.png')

            pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
            pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1)

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                # if iter_num > 0: #use for debug
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_my(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_my(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance2)))
                    save_best = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)