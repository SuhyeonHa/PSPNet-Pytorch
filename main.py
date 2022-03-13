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
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import cv2
from torchvision.utils import save_image

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128],[0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

class SegDataset(Dataset):
    def __init__(self, dataset_dir, crop_size=256, split='train'):
        assert split in ['train', 'trainval', 'val']
        self.dataset_dir = dataset_dir
        # self.transform = transform
        self.crop_size = crop_size
        self.list_path = dataset_dir+'ImageSets/Segmentation/{0}.txt'.format(split)
        self.names = []
        with open(self.list_path, 'r') as file:
            self.names = file.read().splitlines()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        input_path = self.dataset_dir+'JPEGImages/{0}.jpg'.format(self.names[idx])
        target_path = self.dataset_dir+'SegmentationClass/{0}.png'.format(self.names[idx])

        input_img = Image.open(input_path).convert('RGB')
        target_img = np.array(Image.open(target_path).convert('P'))
        # target_img = self._convert_to_segmentation_mask(target_img)

        sample = {
            'input': self.trans(input_img, target_img)[0],
            'target': self.trans(input_img, target_img)[1]
        }
        return sample

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        # https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return

    def trans(self, input, target):
        random_seed=np.random.randint(2010147)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input_transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 2.0)),
                            transforms.RandomRotation(10),
                            transforms.GaussianBlur(3),
                            transforms.ToTensor(),
                            normalize])

        target_transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                             transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 2.0)),
                                             transforms.RandomRotation(10),
                                             transforms.GaussianBlur(3),
                                             transforms.ToTensor()])
        input = input_transform(input)
        target = target_transform(Image.fromarray(target))
        return input, target

    # def validate_image(self, img):
    #     img = np.array(img, dtype=float)
    #     if len(img.shape) < 3:
    #         rgb = np.empty((64, 64, 3), dtype=np.float32)
    #         rgb[:, :, 0] = img
    #     #         rgb[:, :, 1] = img
    #     #         rgb[:, :, 2] = img
    #         img = rgb
    #     return img.transpose(2, 0, 1)

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
        self.num_class = 150
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
        return x, x_aux # torch.Size([b, num_class, h, w])

def save_checkpoint(net, dir_path, epoch):
    path = os.path.abspath(dir_path)
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(net.state_dict(), '{0}/model_{1}.pth'.format(path, epoch))

def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


# configuration
batch_size = 8
num_workers = 0
dataset_dir = '/home/cvmlserver4/suhyeon/dataset/VOCdevkit/VOC2012/'
output_dir = './results'
image_size = 256
lambda_aux = 0.4
num_epochs = 200
lr = 1e-2
power = 0.9
momentum = 0.9
weight_decay = 1e-4
save_model_interval = 50
test_interval = 10

# move the input and model to GPU for speed if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pspnet = PSPNet((image_size, image_size)).to(device)
optimizer = torch.optim.SGD(pspnet.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay)

dataset = SegDataset(dataset_dir, crop_size=image_size, split='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
data_iter = iter(dataloader)
sample = next(data_iter)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.imshow(torchvision.utils.make_grid(sample['input'], normalize=True).permute(1,2,0))
ax1.set_title('SOURCE')
ax1.axis("off")
ax2 = fig.add_subplot(2, 1, 2)
ax2.imshow(torchvision.utils.make_grid(sample['target']).permute(1,2,0))
ax2.set_title('SEGMENTATION')
ax2.axis("off")
plt.show()

val_dataset = SegDataset(dataset_dir, crop_size=image_size, split='trainval')
rand_idx = np.random.randint(len(val_dataset))
val_data = val_dataset[rand_idx]

criterion = nn.CrossEntropyLoss().to(device)

train_loss = []
ce_loss = []
aux_loss = []

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        input = data['input'].to(device)
        target = data['target'].to(device)
        out, aux = pspnet(input)
        loss1 = criterion(out, target)
        loss2 = criterion(aux, target)
        loss = loss1 + lambda_aux * loss2
        ce_loss.append(loss1.item())
        aux_loss.append(loss2.item())
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % test_interval == 0:
            print('[%d/%d][%d/%d]\tLoss_CE: %.4f\tLoss_AUX: %.4f\tloss_Train: %.4f'
                  # 'D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     ce_loss[-1], aux_loss[-1], loss[-1]))

        if (epoch + 1) % save_model_interval == 0 or epoch == num_epochs - 1:
            save_checkpoint(pspnet, epoch + 1)

        if (epoch + 1) % test_interval == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                val_out, _ = pspnet(val_data['input'].to(device))
                val_tgt = val_data['target']
                output = torch.cat((val_out, val_tgt), dim=0)
                output = denorm(output.cpu())
                output_name = os.path.abspath(output_dir) + '/output_epoch_{:0}.jpg'.format(epoch)
                save_image(output, str(output_name))

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(ce_loss, label="CE")
        plt.plot(aux_loss, label="AUX")
        plt.plot(train_loss, label="TRAIN")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        # plt.show()
        fig_name = os.path.abspath(output_dir) + '/plot_epoch_{:0}.jpg'.format(epoch)
        plt.savefig(fig_name)

        # # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output.shape)
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        # probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # print(probabilities)
