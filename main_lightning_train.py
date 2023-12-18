import argparse
import random
import numpy as np
import string
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as L 
import torch
import time

from models.module import TrainingModule

def get_parser():
    parser = argparse.ArgumentParser('SlotCon')

    # dataset
    parser.add_argument('--dataset', type=str, default='COCO', choices=['COCO', 'COCOplus', 'ImageNet'], help='dataset type')
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--image-size', type=int, default=224, help='image crop size')
    parser.add_argument('--min-scale', type=float, default=0.08, help='minimum crop scale')
   
    # model
    parser.add_argument('--arch', type=str, default='resnet50', help='backbone architecture')
    parser.add_argument('--dim-hidden', type=int, default=4096, help='hidden dimension')
    parser.add_argument('--dim-out', type=int, default=256, help='output feature dimension')
    parser.add_argument('--num-prototypes', type=int, default=256, help='number of prototypes')
    parser.add_argument('--teacher-momentum', default=0.99, type=float, help='momentum value for the teacher model')
    parser.add_argument('--teacher-temp', default=0.07, type=float, help='teacher temperature')
    parser.add_argument('--student-temp', default=0.1, type=float, help='student temperature')
    parser.add_argument('--center-momentum', default=0.9, type=float, help='momentum for the center')
    parser.add_argument('--group-loss-weight', default=0.5, type=float, help='balancing weight of the grouping loss')

    # optim.
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size')
    parser.add_argument('--base-lr', type=float, default=1.0,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='sgd', help='optimizer choice')
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--fp16', action='store_true', default=True, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
    
    # misc
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers per GPU to use')
    parser.add_argument('--gpus', type=int, default=8, help='num of GPUs to use')
    parser.add_argument('--nodes', type=int, default=1, help='num of nodes to use')
    parser.add_argument('--strategy', type=str, default="ddp", help='training strategy to use [ddp, fsdp, ...]')

    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = get_parser()

    # TODO : Check args related to distributed training
    args.batch_size = int(args.batch_size / (args.gpus*args.nodes))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ### Create module
    module = TrainingModule(args)

    ### Setup logger 
    letters = string.ascii_lowercase
    run_name = "".join(random.choice(letters) for i in range(8))
    wandb_logger = WandbLogger(project="SlotCon", offline=True, name=run_name)

    ### Monitor learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    ### Prepare the checkpointing
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        dirpath=f"./checkpoints/slotcon_train",
        filename="{epoch:02d}-{train_loss:.2f}"
    )

    # TODO : resume from a checkpoint
    if args.resume:
        pass 

    #########################
    ### Training
    #########################
    start = time.time()
    
    # TODO : prepare trainer args
    trainer_args = {
        "max_epochs": args.epochs,
        "callbacks": [lr_monitor, checkpoint_callback],
        "logger": wandb_logger,
        "precision": 16 if args.fp16 else 32,
        "accelerator": "gpu",
        "devices": args.gpus,
        "num_nodes": args.nodes,
        "strategy": args.strategy,
        "num_sanity_val_steps": 1,
        "check_val_every_n_epoch": 10
    }
    trainer = L.Trainer(**trainer_args)
    trainer.fit(module)

    # Save trained model
    save_path = args.checkpoint_path if args.checkpoint_path is not None else "last_model.pth"
    trainer.save_checkpoint(save_path)

    end = time.time()
    elapsed = end-start
    print(f"Training time {elapsed/60:.2f} min")
    print(f"Memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    print(f"Memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")