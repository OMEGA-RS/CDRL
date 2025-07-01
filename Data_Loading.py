import torch
from torch import nn
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms


class Trans_Dataset(Dataset):
    # Construct Training and Testing Datasets for Trans_Net Training

    def __init__(self, im1, im2, mask, patch_size, isTrain):

        self.im1 = torch.from_numpy(im1).float()
        self.im2 = torch.from_numpy(im2).float()
        self.mask = torch.from_numpy(mask).float()

        self.patch_size = patch_size
        self.isTrain = isTrain

        self.transform_ = transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip()
        ])

        if self.im1.ndimension() == 2:
            self.im1 = self.im1.unsqueeze(0)
        else:
            self.im1 = self.im1.permute(2, 0, 1)

        self.im2 = self.im2.permute(2, 0, 1)
        self.mask = self.mask.unsqueeze(0)

        self.im1_channel = self.im1.shape[0]
        self.im2_channel = self.im2.shape[0]
        self.mask_channel = self.mask.shape[0]

        # Channel splicing
        im = torch.cat([self.im1, self.im2, self.mask], dim=0).unsqueeze(0)  # [1, c1+c2+c3, h, w]

        # Train and test data
        if self.isTrain:

            sub_net = nn.Sequential(
                nn.ReflectionPad2d(
                    (
                        self.patch_size // 2,
                        self.patch_size // 2 - 1,
                        self.patch_size // 2,
                        self.patch_size // 2 - 1,
                    )
                ),

                nn.Unfold(kernel_size=self.patch_size, stride=40)
            )

            self.im_patches = sub_net(im)

            # Dimension conversion
            self.im_patches = rearrange(
                self.im_patches,
                "b (c p1 p2) n -> (b n) c p1 p2",
                p1=self.patch_size,
                p2=self.patch_size,
                c=(self.im1_channel + self.im2_channel + self.mask_channel),
            )

        else:
            self.im_patches = im

    def __getitem__(self, index):
        cur_patch = self.im_patches[index]
        if self.isTrain:
            cur_patch = self.transform_(cur_patch.unsqueeze(0))[0]

        return (cur_patch[: self.im1_channel],
                cur_patch[self.im1_channel: self.im1_channel + self.im2_channel],
                cur_patch[self.im1_channel + self.im2_channel:]
                )

    def __len__(self):
        return len(self.im_patches)


class CD_Dataset(Dataset):
    # Construct Training and Testing Datasets for CD_Net Training

    def __init__(self, x_a, x_b, x_ab, x_ba, pre_map, patch_size, isTrain):

        self.x_a = torch.from_numpy(x_a).float()
        self.x_b = torch.from_numpy(x_b).float()
        self.x_ab = x_ab.squeeze(0).cpu()
        self.x_ba = x_ba.squeeze(0).cpu()

        self.patch_size = patch_size
        self.isTrain = isTrain

        self.transform_ = transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip()
        ])

        # Add dimension
        if self.x_a.ndimension() == 2:
            self.x_a = self.x_a.unsqueeze(0)
        else:
            self.x_a = self.x_a.permute(2, 0, 1)
        self.x_b = self.x_b.permute(2, 0, 1)

        self.x_a_channel = self.x_a.shape[0]
        self.x_b_channel = self.x_b.shape[0]
        self.x_ab_channel = self.x_ab.shape[0]
        self.x_ba_channel = self.x_ba.shape[0]

        # Channel splicing
        im = torch.cat([self.x_a, self.x_b, self.x_ab, self.x_ba], dim=0).unsqueeze(0)  # [1, c1+c2, h, w]

        # Image tiling
        if self.patch_size % 2 != 0:
            sub_net = nn.Sequential(
                      nn.ReflectionPad2d(self.patch_size // 2),
                      nn.Unfold(kernel_size=self.patch_size, stride=1),
            )
        else:
            sub_net = nn.Sequential(
                nn.ReflectionPad2d(
                    (
                        self.patch_size // 2,
                        self.patch_size // 2 - 1,
                        self.patch_size // 2,
                        self.patch_size // 2 - 1,
                    )
                ),
                nn.Unfold(kernel_size=self.patch_size, stride=1)
            )

        self.im_patches = sub_net(im)

        # Dimension conversion
        self.im_patches = rearrange(
            self.im_patches,
            "b (c p1 p2) n -> (b n) c p1 p2",
            p1=self.patch_size,
            c=(self.x_a_channel + self.x_b_channel + self.x_ab_channel + self.x_ba_channel),
        )

        # Select unchanged and changed positions
        self.pre_map = torch.from_numpy(pre_map).float()
        self.pre_map = self.pre_map.flatten()

        if self.isTrain:
            # Select all changed (1) and unchanged (0) pixels
            change_indices = self.pre_map == 1
            un_change_indices = self.pre_map == 0

            # Select the same number of un-change samples as change samples
            num_changes = change_indices.sum().item()
            un_change_indices = un_change_indices.nonzero().squeeze()

            # Randomly sample the same number of no-change samples as the number of change samples
            if len(un_change_indices) > num_changes:
                selected_un_change_indices = np.random.choice(un_change_indices.numpy(), size=num_changes, replace=False)
            else:
                selected_un_change_indices = un_change_indices.numpy()

            selected_un_change_indices = torch.tensor(selected_un_change_indices)

            # Combine change and selected un-change samples
            selected_indices = torch.cat([change_indices.nonzero().squeeze(), selected_un_change_indices])
            self.im_patches = self.im_patches[selected_indices]
            self.label = self.pre_map[selected_indices]

            # Shuffle the dataset
            random_array = np.random.choice(np.arange(0, len(self.label)), size=len(self.label), replace=False)
            self.im_patches = self.im_patches[random_array]
            self.label = self.label[random_array]
        else:
            self.im_patches = self.im_patches[self.pre_map == 2]
            self.label = self.pre_map[self.pre_map == 2]

    def __getitem__(self, index):
        cur_patch = self.im_patches[index]
        cur_label = self.label[index]
        if self.isTrain:
            cur_patch = self.transform_(cur_patch.unsqueeze(0))[0]

        return cur_patch[: self.x_a_channel], cur_patch[self.x_a_channel:self.x_a_channel+self.x_b_channel], cur_patch[self.x_a_channel+self.x_b_channel:self.x_a_channel+self.x_b_channel+self.x_ab_channel], cur_patch[self.x_a_channel+self.x_b_channel+self.x_ab_channel:], cur_label

    def __len__(self):
        return len(self.im_patches)
