import os
import torch
import numpy as np
from datetime import datetime
import time
import pandas as pd
from glob import glob
#from apex import amp


class AverageMeter(object):
    """ Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TorchFitter:

    def __init__(self,
                 model,
                 device,
                 loss,
                 optimizer=None,
                 lr=0.0001,
                 scheduler_class=None,
                 scheduler_params=None,
                 folder='models',
                 verbose=0,
                 validation_scheduler=True,
                 step_scheduler=False):

        if type(loss) == type:
            self.loss_function = loss()
        else:
            self.loss_function = loss

        self.epoch = 0  # current epoch
        self.verbose = verbose

        self.base_dir = f'./{folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_metric = 0

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # Optimizer object
        if not optimizer:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler_class(self.optimizer, **scheduler_params)

        self.validation_scheduler = validation_scheduler  # do scheduler.step after validation stage loss
        self.step_scheduler = step_scheduler  # do scheduler.step after optimizer.step
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader, n_epochs=1, early_stopping=0, metric=None, metric_kwargs=None, early_stopping_mode='min'):
        """ Fits a model

        Inputs:
            train_loader: Training data
            validation_loader: Validation Data

        Outputs:
            returns a pandas DataFrame object with training history
        """
        self.best_metric = 10 ** 5 if early_stopping_mode == 'min' else -10 ** 5

        training_history = []
        es_epochs = 0
        for e in range(n_epochs):
            history = {'epoch': e}  # training history log for this epoch

            # Update log
            if self.verbose > 0:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            # Run one training epoch
            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)
            history['train'] = summary_loss.avg  # training loss

            # Print training result
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, '
                     f'time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            # Run epoch validation
            t = time.time()
            summary_loss, calculated_metric = self.validation(validation_loader, metric, metric_kwargs)
            history['val'] = summary_loss.avg  # validation loss
            history['lr'] = self.optimizer.param_groups[0]['lr']

            # Print validation results
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, ' +\
                f'metric {calculated_metric},' if calculated_metric else '' +\
                f'time: {(time.time() - t):.5f}')

            if calculated_metric:
                history['val_metric'] = calculated_metric

            # Check if result is improved, then save model
            calculated_metric = calculated_metric if calculated_metric else summary_loss.avg
            if (((metric) and
                    (((early_stopping_mode == 'max') and (calculated_metric > self.best_metric)) or
                    ((early_stopping_mode == 'min') and (calculated_metric < self.best_metric))))
                or
                ((metric is None) and (calculated_metric < self.best_metric))):
                    savepath = f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin'
                    self.log(f'Validation metric improved from {self.best_metric} to '
                                 f'{calculated_metric} and model is saved to {savepath}')
                    self.best_metric = calculated_metric
                    self.model.eval()
                    self.save(savepath)
                    for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                        os.remove(path)
                    es_epochs = 0  # reset early stopping count
            else:
                es_epochs += 1  # increase epoch count with no improvement, for early stopping check

            # Check if Early Stopping condition is met
            if (early_stopping > 0) & (es_epochs > early_stopping):
                self.log(f'Early Stopping: {early_stopping} epochs with no improvement')
                training_history.append(history)
                break

            if self.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            training_history.append(history)
            self.epoch += 1

        return pd.DataFrame(training_history).set_index('epoch')

    def validation(self, val_loader, metric=None, metric_kwargs=None):
        self.model.eval()
        summary_loss = AverageMeter()
        y_preds = []
        y_true = []

        t = time.time()
        for step, (images, labels) in enumerate(val_loader):
            if self.verbose > 0:
                if step % self.verbose == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, '
                        f'summary_loss: {summary_loss.avg:.5f}, '
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():  # no gradient update
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                labels = labels.to(self.device)

                if metric:
                    arr = labels.cpu().numpy()
                    y_true += np.argmax(arr, axis=1).tolist() if len(arr.shape)==2 else arr.tolist()

                # just forward propagation
                output = self.model(images)
                loss = self.loss_function(output, labels)
                summary_loss.update(loss.detach().item(), batch_size)

                if metric:
                    arr = output.cpu().numpy()
                    y_preds += np.argmax(arr, axis=1).tolist() if len(arr.shape)==2 else arr.tolist()

        calculated_metric = metric(y_true, y_preds, **metric_kwargs) if metric else None

        return summary_loss, calculated_metric

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
            loss.backward()

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
                'best_summary_loss': self.best_metric,
                'epoch': self.epoch,
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
        self.best_metric = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        """ Log training ouput into console and file
            input:
                message: message to be logged
        """
        if self.verbose > 0:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


class TorchFitterBoxes(TorchFitter):

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.verbose:
                if step % self.verbose == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, '
                        f'summary_loss: {summary_loss.avg:.5f}, '
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():  # no gradient update
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                # just forward propagation
                loss, _, _ = self.model(images, boxes, labels)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

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
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.verbose > 0:
                if step % self.verbose == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            # extract images, boxes and labels from the dataloader
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            self.optimizer.zero_grad()

            loss, _, _ = self.model(images, boxes, labels)
            # print(loss)

            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.step_scheduler:
                self.scheduler.step()

        return summary_loss
