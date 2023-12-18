
import lightning.pytorch as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

import torch    
from torch.utils.data import DataLoader

from models import resnet
from models.slotcon import SlotCon
from utils.lars import LARS
from data.datasets import ImageFolder
from data.transforms import CustomDataAugmentation
from utils.lr_scheduler import get_scheduler
from utils.util import AverageMeter 

# TODO : setup validation and testing steps
# TODO : setup ViT training 
class TrainingModule(L.LightningModule):
    def __init__(
            self, 
            args
        ):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = resnet.__dict__[args.arch]
        self.model = SlotCon(self.encoder, args) # directly contains teacher and student networks
        self.automatic_optimization = False
        self.loss_meter = AverageMeter()
    
    def configure_optimizers(self):
        if self.hparams.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.batch_size * self.hparams.world_size / 256 * self.hparams.base_lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'lars':
            optimizer = LARS(
                self.model.parameters(),
                lr=self.hparams.batch_size * self.hparams.world_size / 256 * self.hparams.base_lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay)
        # TODO : check with ViT
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.batch_size * self.hparams.world_size / 256 * self.hparams.base_lr,
                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def setup(
            self, 
            stage: str
        ):
        self.transform = CustomDataAugmentation(
            self.hparams.args.image_size, 
            self.hparams.args.min_scale
        )
        self.train_dataset = ImageFolder(
            self.hparams.args.dataset, 
            self.hparams.args.data_dir, 
            self.transform
        )
        
    def train_dataloader(self): 
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.args.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.args.num_workers, 
            pin_memory=True, 
            drop_last=True
        )     
        # TODO : check the scheduler
        self.scheduler = get_scheduler(
            self.optimizer, 
            len(self.train_dataset) // self.hparams.args.batch_size, 
            self.hparams.args
        )    
        return self.train_loader
    
    def val_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=1, 
            shuffle=True, 
            num_workers=self.hparams.args.num_workers, 
            pin_memory=True, 
        )  

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        crops, coords, flags = batch
        crops = [crop for crop in crops]
        coords = [coord for coord in coords]
        flags = [flag for flag in flags]        
        opt = self.optimizers()

        # compute the loss (forward pass + EMA updates of the teacher network)
        loss = self.model((crops, coords, flags))
        
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.scheduler.step()

        # avg loss from batch size
        self.loss_meter.update(loss.item(), crops[0].size(0))

        # TODO : add logging
        self.log('train_loss', self.loss_meter.val, on_step=True)
        self.log('train_loss_avg', self.loss_meter.avg, on_step=True)
        self.log('lr', self.scheduler.get_lr(), on_step=True)
        # TODO : check if necessary (maybe for ViT) ?
        # self.log('momentum', self.scheduler.get_momentum(), on_step=True)
        # self.log('weight_decay', self.scheduler.get_weight_decay(), on_step=True)
    