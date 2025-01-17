import glob
import numpy as np
import copy

from PIL import Image
from torch.utils.data import Dataset
import lightning.pytorch as L
from torch.utils.data import DataLoader
from .transforms import CustomDataAugmentation

class ImageFolder(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform
    ):
        super(ImageFolder, self).__init__()
        if dataset == 'ImageNet':
            self.fnames = list(glob.glob(data_dir + '/train/*/*.JPEG'))
        elif dataset == 'COCO':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg'))
        elif dataset == 'COCOplus':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg')) + list(glob.glob(data_dir + '/unlabeled2017/*.jpg'))
        elif dataset == 'COCOval':
            self.fnames = list(glob.glob(data_dir + '/val2017/*.jpg'))
        else:
            raise NotImplementedError

        self.fnames = np.array(self.fnames) # to avoid memory leak
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fpath = self.fnames[idx]
        image = Image.open(fpath).convert('RGB')
        return self.transform(image)


# Implementation following : https://github.com/Wuziyi616/SlotDiffusion
class COCOModule(L.LightningDataModule):
    def __init__(self, data_args):
        super().__init__()
        """Build COCO2017 dataset that load images."""
        self.save_hyperparameters()
        self.transform = CustomDataAugmentation(
            data_args.image_size, 
            data_args.min_scale
        )
    
    def setup(self, stage="fit"):
        self.train_dataset = ImageFolder(
            self.hparams.data_args.dataset, 
            self.hparams.data_args.data_dir, 
            self.transform
        )
        self.val_dataset = copy.deepcopy(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.data_args.batch_size,
            shuffle=True,
            num_workers=self.hparams.data_args.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=max(1, self.hparams.data_args.batch_size),
            shuffle=False,  # originally True
            num_workers=self.hparams.data_args.num_workers,
            pin_memory=False,
            drop_last=False
        )