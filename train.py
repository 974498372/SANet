from argparse import ArgumentParser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.SA import SABody
from nets.training import TotalLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import SA_dataset_collate, SADataset
from utils.utils_fit import fit_one_epoch


def parse_args():
    # Setting parameters
    parser = ArgumentParser()

    parser.add_argument('-g','--Cuda', action="store_true", help='if use cuda')
    parser.add_argument('--model-path', type=str, default='', help='The path of weight')
    parser.add_argument('--input-shape', type=int, default=640, help='size of image')
    parser.add_argument('-c','--Cosine-scheduler', action="store_true", help='Cosine_scheduler')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-f','--Freeze', action="store_true", help='Freeze the backbone')
    parser.add_argument('--num-workers', type=int, default=4, help='num_worker')
    parser.add_argument('--train-annotation-path', type=str, default='2007_train.txt', help='path of the train.txt')
    parser.add_argument('--val-annotation-path', type=str, default='2007_train.txt', help='path of the val.txt')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        self.Cuda               = args.Cuda
        model_path              = args.model_path
        input_shape             = [args.input_shape, args.input_shape]
        Cosine_scheduler        = args.Cosine_scheduler
        
        self.epochs             = args.epoch
        self.batch_size         = args.batch_size
        lr                      = args.lr   
        
        
        self.Freeze_Train       = args.Freeze  
        num_workers             = args.num_workers
        train_annotation_path   = args.train_annotation_path
        val_annotation_path     = args.val_annotation_path
        
        # model
        self.model = SABody(num_classes=1)
        weights_init(self.model)
        if model_path != '':
            print('Load weights {}.'.format(model_path))
            device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dict      = self.model.state_dict()
            pretrained_dict = torch.load(model_path, map_location = device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            
        self.model_train = self.model.train()
        
        # GPU
        if self.Cuda:
            self.model_train = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
            self.model_train = self.model_train.cuda()
            
        # data
        with open(train_annotation_path) as f: train_lines = f.readlines()
        with open(val_annotation_path) as f: val_lines   = f.readlines()
        self.num_train = len(train_lines)
        self.num_val = len(val_lines)
        train_dataset = SADataset(train_lines, input_shape, train=True)
        val_dataset = SADataset(val_lines, input_shape, train=False)
        self.gen             = DataLoader(train_dataset, shuffle = True, batch_size = self.batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=SA_dataset_collate)
        self.gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = self.batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=SA_dataset_collate)
            
        # loss
        self.Total_loss = TotalLoss(num_classes=1)
        self.loss_history = LossHistory("logs/")
        
        # optimizer
        self.optimizer = optim.Adam(self.model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_scheduler:
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)
        else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.92)
                
        
                
    def train(self):
        # train
        if self.Freeze_Train:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        epoch_step = self.num_train // self.batch_size
        epoch_step_val = self.num_val // self.batch_size
        if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("dataset is not enough !")
            
        for epoch in range(self.epochs):
            fit_one_epoch(self.model_train, self.model, self.Total_loss, self.loss_history, self.optimizer, epoch, 
                        epoch_step, epoch_step_val, self.gen, self.gen_val, self.epochs, self.Cuda)
            self.lr_scheduler.step()
                
    
if __name__ == "__main__":
    args = parse_args()
    Trainer(args).train()

    