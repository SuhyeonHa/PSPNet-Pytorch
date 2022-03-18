import os
import io
import random

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb
from PIL import Image
import torch
from torch.autograd import Variable
import pdb
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import cv2
from torchvision.utils import save_image
import transform

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128],[0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


class SegDataset(Dataset):
    def __init__(self, dataset_dir, crop_size=256, split='train', transform=None):
        assert split in ['train', 'trainval', 'val']
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.crop_size = crop_size
        self.list_path = dataset_dir+'ImageSets/Segmentation/{0}.txt'.format(split)
        self.names = []
        with open(self.list_path, 'r') as file:
            self.names = file.read().splitlines()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        input_path = self.dataset_dir+'JPEGImages/{0}.jpg'.format(self.names[idx])
        label_path = self.dataset_dir+'SegmentationClass/{0}.png'.format(self.names[idx])

        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        label = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = self.process_mask(label, VOC_COLORMAP)
        label = np.argmax(label, axis=0)

        if self.transform is not None:
            image, label = self.transform(image, label)
            label[label == 255] = 0
        return image, label

    def process_mask(self, rgb_mask, colormap):
        output_mask = []

        for i, color in enumerate(colormap):
            cmap = np.all(np.equal(rgb_mask, color), axis=-1)
            cv2.imwrite(f'mask/{i}.png', cmap*255)
            output_mask.append(cmap)

        return output_mask


class PPM(nn.Module):
    def __init__(self, int_size, in_dim, bins):
        super(PPM, self).__init__()
        self.int_size = int_size
        self.num_level = len(bins)
        self.levels = []
        for bin in bins:
            self.levels.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, in_dim//self.num_level, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_dim//self.num_level),
                nn.ReLU(inplace=True)
            ))
        self.levels = nn.ModuleList(self.levels)

    def forward(self, x):
        out = [x]
        for f in self.levels:
            out.append(F.interpolate(f(x), self.int_size, mode='bilinear'))
        out = torch.cat(out, dim=1)
        return out

class PSPNet(nn.Module):
    def __init__(self, img_size, dropout = 0.1):
        super(PSPNet, self).__init__()
        self.dim = 1024
        # self.classes = 150 # ADE20K
        self.img_size = img_size
        self.int_size = (img_size[0]//8, img_size[1]//8)
        self.num_class = 21
        self.ppm = PPM(self.int_size, self.dim, (1,2,3,6))
        self.resnet = torchvision.models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.layer1, self.layer2 = self.resnet.layer1, self.resnet.layer2
        self.layer3, self.layer4 = self.resnet.layer3, self.resnet.layer4
        self.classifier = nn.Sequential(
            nn.Conv2d(self.dim*2, self.dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(self.dim, self.num_class, kernel_size=1)
        )
        self.aux = nn.Sequential(
            nn.Conv2d(self.dim*2, self.dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(self.dim, self.num_class, kernel_size=1)
        )

    def forward(self, x):
        # torch.Size([b, 3, h, w]) :256
        x = self.layer0(x)
        # torch.Size([b, 64, h/4, w/4])
        x = self.layer1(x)
        # torch.Size([b, 256, h/4, w/4])
        x = self.layer2(x)
        # torch.Size([b, 512, h/8, w/8])
        x_3 = self.layer3(x)
        # torch.Size([b, 1024, h/8, w/8])
        x = self.ppm(x_3)
        x = self.classifier(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear')

        x_aux = self.layer4(x_3)
        # torch.Size([b, 2048, h/8, w/8])
        x_aux = self.aux(x_aux) # torch.Size([b, num_class, h/8, w/8])
        x_aux = F.interpolate(x_aux, size=self.img_size, mode='bilinear')
        # x = torch.argmax(x, dim=1)
        # x_aux = torch.argmax(x_aux, dim=1)
        return x, x_aux # torch.Size([b, num_class, h, w])

def save_checkpoint(net, optimizer, scheduler, dir_path):
    path = os.path.abspath(dir_path)
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(net.state_dict(), '{0}/model.pth'.format(path))
    torch.save(optimizer.state_dict(), '{0}/optimizer.pth'.format(path))
    torch.save(scheduler.state_dict(), '{0}/scheduler.pth'.format(path))

def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(output, 1)

    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union

def poly_learning_rate(base_lr, iter, max_iter, power):
    new_lr = base_lr * (1-float(iter) / max_iter) ** power
    return new_lr

class Trainer(object):
    def __init__(self):
        train_transform = transform.Compose([
            transform.RandScale([0.5, 2]),
            transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([image_size, image_size], crop_type='rand', padding=mean, ignore_label=255),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        val_transform = transform.Compose([
            transform.Crop([image_size, image_size], crop_type='center', padding=mean, ignore_label=255),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        dataset = SegDataset(dataset_dir, crop_size=image_size, split='train', transform=train_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        data_iter = iter(dataloader)
        # sample = next(data_iter)

        val_dataset = SegDataset(dataset_dir, crop_size=image_size, split='trainval', transform=val_transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_data_iter = iter(val_dataloader)

        # val_dataset = SegDataset(dataset_dir, crop_size=image_size, split='trainval')
        # rand_idx = np.random.randint(len(val_dataset))
        # val_data = val_dataset[rand_idx]

        criterion = nn.CrossEntropyLoss().to(device)

        torch.cuda.empty_cache()
        train_losses = []; val_losses = []
        train_mIoUs = []; train_pixAccs = []
        val_mIoUs = []; val_pixAccs = []

        for epoch in range(num_epochs):
            # training #
            total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
            total_loss, total_mIoU, total_pixAcc = 0, 0, 0
            for i, data in enumerate(dataloader, 0):
                input = data[0].to(device)
                target = data[1].to(device)
                out, aux = pspnet(input)
                loss1 = criterion(out, target)
                loss2 = criterion(aux, target)
                train_loss = loss1 + lambda_aux * loss2

                # evaluation metrics
                correct, labeled = batch_pix_accuracy(out, target)
                inter, union = batch_intersection_union(out, target, num_class)

                total_correct += correct
                total_label += labeled
                total_inter += inter
                total_union += union
                train_pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                train_mIoU = IoU.mean()

                total_loss += train_loss.item()
                total_pixAcc += train_pixAcc.item()
                total_mIoU += train_mIoU.item()

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                scheduler.step()

            train_losses.append(total_loss / len(dataloader))
            train_pixAccs.append(total_pixAcc / len(dataloader))
            train_mIoUs.append(total_mIoU / len(dataloader))

            # validation #
            # if (epoch + 1) % test_interval == 0 or epoch == num_epochs - 1:
            pspnet.eval()
            total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
            total_loss, total_mIoU, total_pixAcc = 0, 0, 0
            for i, data in enumerate(dataloader, 0):
                with torch.no_grad():
                    val_input = data[0].to(device)
                    val_target = data[1].to(device)

                    val_out, _ = pspnet(val_input)
                    val_loss = criterion(val_out, val_target)

                    correct, labeled = batch_pix_accuracy(val_out, val_target)
                    inter, union = batch_intersection_union(val_out, val_target, num_class)

                    total_correct += correct
                    total_label += labeled
                    total_inter += inter
                    total_union += union
                    val_pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                    val_IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                    val_mIoU = val_IoU.mean()

                    total_loss += val_loss.item()
                    total_pixAcc += val_pixAcc.item()
                    total_mIoU += val_mIoU.item()

            val_losses.append(total_loss / len(val_dataloader))
            val_pixAccs.append(total_pixAcc / len(val_dataloader))
            val_mIoUs.append(total_mIoU / len(val_dataloader))

            # if epoch % test_interval == 0:
            print('[%d/%d][%d]\tLoss_Train: %.4f\tLoss_Val: %.4f'
                  '\tmIoU_Train: %.4f''\tmIoU_Val: %.4f''\tAcc_Train: %.4f\tAcc_Val: %.4f'
                  % (epoch, num_epochs, len(dataloader),
                     train_losses[-1], val_losses[-1],
                     train_mIoUs[-1], val_mIoUs[-1],
                     train_pixAccs[-1], val_pixAccs[-1]))

            if (epoch + 1) % save_model_interval == 0 or epoch == num_epochs - 1:
                save_checkpoint(pspnet, optimizer, scheduler, 'checkpoints')

            fig = plt.figure()
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(val_losses, label='val')
            ax1.plot(train_losses, label='train')
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend(), ax1.grid()
            ax1.set_title('Loss')

            ax2 = fig.add_subplot(3, 1, 2)
            ax2.plot(val_mIoUs, label='val')
            ax2.plot(train_mIoUs, label='train')
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("mIoU")
            ax2.legend(), ax2.grid()
            ax2.set_title('Score per epoch')

            ax3 = fig.add_subplot(3, 1, 3)
            ax3.plot(val_pixAccs, label='val')
            ax3.plot(train_pixAccs, label='train')
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Accuracy")
            ax3.legend(), ax3.grid()
            ax3.set_title('Accuracy per epoch')

            # plt.show()
            fig_name = os.path.abspath(output_dir) + '/plot_epoch_{:0}.jpg'.format(epoch)
            plt.savefig(fig_name)

class Tester(object):
    def __init__(self):
        epoch = 500
        val_transform = transform.Compose([
            # transform.Crop([image_size, image_size], crop_type='center', padding=mean, ignore_label=255),
            transform.Resize([image_size, image_size]),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        criterion = nn.CrossEntropyLoss().to(device)

        torch.cuda.empty_cache()
        losses = []
        mIoUs = []
        pixAccs = []

        dataset = SegDataset(dataset_dir, crop_size=image_size, split='val', transform=val_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        pspnet.load_state_dict(torch.load('./checkpoints/'+'model.pth'))
        optimizer.load_state_dict(torch.load('./checkpoints/'+'optimizer.pth'))

        pspnet.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        total_loss, total_mIoU, total_pixAcc = 0, 0, 0
        for i, data in enumerate(dataloader, 0):
            with torch.no_grad():
                input = data[0].to(device)
                target = data[1].to(device)

                val_out, _ = pspnet(input)
                val_loss = criterion(val_out, target)

                correct, labeled = batch_pix_accuracy(val_out, target)
                inter, union = batch_intersection_union(val_out, target, num_class)

                total_correct += correct
                total_label += labeled
                total_inter += inter
                total_union += union
                val_pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                val_IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                val_mIoU = val_IoU.mean()

                total_loss += val_loss.item()
                total_pixAcc += val_pixAcc.item()
                total_mIoU += val_mIoU.item()

                # image save
                pred = val_out.squeeze().argmax(dim=0).cpu()
                gray = np.uint8(pred)
                color = Image.fromarray(gray.astype(np.uint8)).convert('P')
                color.putpalette(np.array(VOC_COLORMAP).astype('uint8'))
                # gray_path = os.path.join('./results/testset/' + '{0}_gray.png').format(i)
                color_path = os.path.join('./results/testset/' + '{0}_seg.png').format(i)
                # cv2.imwrite(gray_path, gray)
                color.save(color_path)

        losses.append(total_loss / len(dataloader))
        pixAccs.append(total_pixAcc / len(dataloader))
        mIoUs.append(total_mIoU / len(dataloader))

        print('[%d][%d]\tLoss_test: %.4f''\tmIoU_test: %.4f''\tAcc_test: %.4f'
              % (epoch, len(dataloader),
                 losses[-1], mIoUs[-1], pixAccs[-1]))

# configuration
batch_size = 1
num_workers = 0
dataset_dir = '/home/cvmlserver4/suhyeon/dataset/VOCdevkit/VOC2012/'
output_dir = './results'
image_size = 256
lambda_aux = 0.4
num_epochs = 30000
lr = 1e-2
power = 0.9
momentum = 0.9
weight_decay = 1e-4
save_model_interval = 500
test_interval = 100
num_class = 21

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

# move the input and model to GPU for speed if available
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
pspnet = PSPNet((image_size, image_size)).to(device)
optimizer = torch.optim.SGD(pspnet.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
     lr_lambda = lambda step: poly_learning_rate(base_lr=lr, max_iter=num_epochs, iter=0, power=power))

tester = Tester()

