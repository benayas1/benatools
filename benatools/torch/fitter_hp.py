import os
import torch
from datetime import datetime
import time
import pandas as pd
from glob import glob
from apex import amp
from .fitter import TorchFitter, AverageMeter


class TorchFitterHP(TorchFitter):

    def __init__(self,
                 model,
                 device,
                 loss,
                 n_epochs=1,
                 optimizer=None,
                 lr=0.0001,
                 scheduler_class=None,
                 scheduler_params=None,
                 folder='models',
                 verbose=0,
                 validation_scheduler=True,
                 step_scheduler=False,
                 early_stopping=0,
                 opt_level='O1'):
        super(TorchFitterHP, self).__init__(model,
                                            device,
                                            loss,
                                            n_epochs,
                                            optimizer,
                                            lr,
                                            scheduler_class,
                                            scheduler_params,
                                            folder,
                                            verbose,
                                            validation_scheduler,
                                            step_scheduler,
                                            early_stopping)
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)

    def train_one_epoch(self, train_loader):
        """ Run one epoch on the train dataset
            inputs:
                train_loader: DataLoader containing the training dataset
            outputs:
                summary_loss: AverageMeter object with this epochs's average loss
        """
        self.model.train()  # set train mode
        summary_loss = AverageMeter()  # object to track the average loss
        t = time.time()

        # run epoch
        for step, (images, labels) in enumerate(train_loader):
            if self.verbose > 0:
                if step % self.verbose == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            # extract images and labels from the dataloader
            batch_size = images.shape[0]
            images = images.to(self.device).float()
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Output and loss
            output = self.model(images)
            loss = self.loss_function(output, labels)

            # backpropagation
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.step_scheduler:
                self.scheduler.step()

        return summary_loss

    def save(self, path):
        """ Save model and other metadata
        input:
            path: path of the file to be saved
        """
        self.model.eval()
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_summary_loss': self.best_summary_loss,
                'epoch': self.epoch,
                'amp': amp.state_dict()
        }, path)

    def load(self, path):
        """ Load model and other metadata
            input:
                path: path of the file to be loaded
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        amp.load_state_dict(checkpoint['amp'])