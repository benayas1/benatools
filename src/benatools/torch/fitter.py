import os
import torch
import numpy as np
from datetime import datetime
import time
import pandas as pd


class AverageMeter(object):
    """
    Computes and stores the average and current value

    Attributes
    ----------
    val : float
        Stores the average loss of the last batch
    avg : float
        Average loss
    sum : float
        Sum of all losses
    count : int
        number of elements
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
        """
        Updates current internal state

        Parameters
        ----------
        val : float
            loss on each training step
        n : int, Optional
            batch size
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TorchFitter:
    """
    Helper class to implement a training loop in PyTorch

    Parameters
    ----------
    model : torch.nn.Module
        Model to be fitted
    device : int
        device can be cuda or cpu
    loss : torch.nn.loss object or function returning
        DataFrame to split
    optimizer : torch.optim object
        Optimizer object
    schedule :
        Scheduler object
    validation schedule :
        Scheduler object for the validation step
    step_scheduler=False:
    folder : str
        Optional, folder where to store checkpoints
    verbose : bool
        Whether to print outputs or not
    log_file : bool
        whether to write the log in log.txt or not
    """

    def __init__(self,
                 model,
                 device,
                 loss,
                 optimizer,
                 scheduler=None,
                 validation_scheduler=True,
                 step_scheduler=False,
                 folder='models',
                 verbose=True,
                 log_file=True,
                 ):

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

        # Optimizer object
        self.optimizer = optimizer

        # Scheduler Object
        self.scheduler = scheduler
        self.validation_scheduler = validation_scheduler  # do scheduler.step after validation stage loss
        self.step_scheduler = step_scheduler  # do scheduler.step after optimizer.step
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, 
            train_loader, 
            val_loader, 
            n_epochs=1, 
            metric=None, 
            metric_kwargs={}, 
            early_stopping=0, 
            early_stopping_mode='min', 
            save_checkpoint=True,
            verbose_steps=0):
        """ Fits a model

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Training data
        val_loader : torch.utils.data.DataLoader
            Validation Data
        n_epochs : int
            Maximum number of epochs to train
        metric : function with (y_true, y_pred, **metric_kwargs) signature
            Metric to evaluate results on
        metric_kwargs : dict
            Arguments for the passed metric. Ignored if metric is None
        early_stopping : int
            Early stopping epochs
        early_stopping_mode : str
            Min or max criteria
        save_checkpoint : bool
            Whether to save the checkpoint when training
        verbose_steps : int, defaults to 0
            number of step to print every training summary

        Returns
        -------
        returns a pandas DataFrame object with training history
        """
        self.best_metric = 10 ** 7 if early_stopping_mode == 'min' else -10 ** 7

        training_history = []
        es_epochs = 0
        for e in range(n_epochs):
            history = {'epoch': e}  # training history log for this epoch

            # Update log
            lr = self.optimizer.param_groups[0]['lr']
            self.log(f'\n{datetime.utcnow().isoformat()}\nEPOCH {str(self.epoch+1)}/{str(n_epochs)} - LR: {lr}')

            # Run one training epoch
            t = time.time()
            train_summary_loss = self.train_one_epoch(train_loader, verbose_steps=verbose_steps)
            history['train'] = train_summary_loss.avg  # training loss

            # Save checkpoint
            if save_checkpoint:
                self.save(f'{self.base_dir}/last-checkpoint.bin', False)

            # Run epoch validation
            val_summary_loss, calculated_metric = self.validation(val_loader, metric=metric, metric_kwargs=metric_kwargs, verbose_steps=verbose_steps)
            history['val'] = val_summary_loss.avg  # validation loss
            history['lr'] = self.optimizer.param_groups[0]['lr']

            metric_log = f'- metric {calculated_metric},' if calculated_metric else ''
            self.log(f'\r[RESULT] {(time.time() - t):.2f}s - train loss: {train_summary_loss.avg:.5f} - val loss: {val_summary_loss.avg:.5f} ' + metric_log)

            if calculated_metric:
                history['val_metric'] = calculated_metric

            # Check if result is improved, then save model
            calculated_metric = calculated_metric if calculated_metric else val_summary_loss.avg
            if (((metric) and
                    (((early_stopping_mode == 'max') and (calculated_metric > self.best_metric)) or
                    ((early_stopping_mode == 'min') and (calculated_metric < self.best_metric))))
                or
                ((metric is None) and (calculated_metric < self.best_metric))):
                    self.log(f'Validation metric improved from {self.best_metric} to {calculated_metric}')
                    self.best_metric = calculated_metric
                    self.model.eval()
                    if save_checkpoint:
                        savepath = f'{self.base_dir}/best-checkpoint.bin'
                        self.save(savepath)
                    es_epochs = 0  # reset early stopping count
            else:
                es_epochs += 1  # increase epoch count with no improvement, for early stopping check

            # Check if Early Stopping condition is met
            if (early_stopping > 0) & (es_epochs > early_stopping):
                self.log(f'Early Stopping: {early_stopping} epochs with no improvement')
                training_history.append(history)
                break

            # Scheduler step after validation
            if self.validation_scheduler:
                self.scheduler.step(metrics=calculated_metric)

            training_history.append(history)
            self.epoch += 1

        return pd.DataFrame(training_history).set_index('epoch')

    def train_one_epoch(self, train_loader, verbose_steps=0):
        """
        Run one epoch on the train dataset

        Parameters
        ----------
        train_loader : torch.DataLoader
            DataLoaders containing the training dataset
        verbose_steps : int, defaults to 0
            number of step to print every training summary

        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        """
        self.model.train()  # set train mode
        summary_loss = AverageMeter()  # object to track the average loss
        t = time.time()

        # run epoch
        for step, data in enumerate(train_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rTrain Step {step}/{len(train_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.2f} secs, ' +
                        f'ETA: {(len(train_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            # Calculate batch size
            batch_size = data['x'].shape[0]

            # Run one batch
            loss = self.train_one_batch(data)

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.step_scheduler:
                self.scheduler.step()

        self.log(f'\r[TRAIN] {(time.time() - t):.2f}s - train loss: {summary_loss.avg:.5f}')

        return summary_loss

    def train_one_batch(self, data):
        """
        Trains one batch of data. 
        The actions to be done here are: 
        - extract x and y (labels)
        - calculate output and loss
        - backpropagate

        Parameters
        ----------
        data : Tuple of torch.Tensor
            Batch of data. Normally the first element is the data (X) and the second is the label (y), but it dependes on what the Dataset outputs.

        Returns
        -------
        torch.Tensor
            A tensor with the calculated loss
        int
            The batch size 
        """
        # extract x and y from the dataloader
        x = data['x']
        y = data['y']

        # send them to device
        x = x.to(self.device).float()
        y = y.to(self.device)

        self.optimizer.zero_grad()

        # Output and loss
        output = self.model(x)
        loss = self.loss_function(output, y)

        # backpropagation
        loss.backward()

        return loss

    def validation(self, val_loader, metric=None, metric_kwargs={}, verbose_steps=0):
        """
        Validates a model

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            Validation Data
        metric : function with (y_true, y_pred, **metric_kwargs) signature
            Metric to evaluate results on
        metric_kwargs : dict
            Arguments for the passed metric. Ignored if metric is None
        verbose_steps : int, defaults to 0
            number of step to print every training summary

        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        float
            Calculated metric if a metric is provided, else None
        """

        self.model.eval()
        summary_loss = AverageMeter()
        y_preds = []
        y_true = []

        t = time.time()
        for step, data in enumerate(val_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rVal Step {step}/{len(val_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.2f} secs,' + 
                        f'ETA: {(len(val_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                x = data['x'].to(self.device).float()
                y = data['y'].to(self.device)
                batch_size = x.shape[0]

                if metric:
                    y_true += y.cpu().numpy().tolist()

                # just forward propagation
                output = self.model(x)
                loss = self.loss_function(output, y)
                summary_loss.update(loss.detach().item(), batch_size)

                if metric:
                    y_preds += output.cpu().numpy().tolist()

        calculated_metric = metric(y_true, y_preds, **metric_kwargs) if metric else None
        metric_log = f'- metric {calculated_metric},' if calculated_metric else ''
        self.log(f'\r[VALIDATION] {(time.time() - t):.2f}s - val. loss: {summary_loss.avg:.5f} ' + metric_log)
        return summary_loss, calculated_metric

    def predict(self, test_loader, verbose_steps=0):
        """
        Makes predictions using the trained model

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Test Data
        verbose_steps : int, defaults to 0
            number of step to print every training summary    

        Returns
        -------
        np.array
            Predicted values by the model
        """

        self.model.eval()
        y_preds = []
        t = time.time()

        for step, data in enumerate(test_loader):
            if self.verbose & (verbose_steps > 0) > 0:
                if step % verbose_steps == 0:
                    print(
                        f'\Prediction Step {step}/{len(test_loader)}, ' +
                        f'time: {(time.time() - t):.2f} secs,' + 
                        f'ETA: {(len(test_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                x = data['x'].to(self.device).float()
                batch_size = x.shape[0]

                # Output and loss
                output = self.model(x)

                y_preds += output.cpu().numpy().tolist()

        return np.array(y_preds)

    def save(self, path, verbose=True):
        """
        Save model and other metadata

        Parameters
        ----------
        path : str
            Path of the file to be saved
        verbose : int
            1 = print logs, 0 = silence
        """
        if verbose:
            self.log(f'Model is saved to {path}')
        self.model.eval()

        data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_summary_loss': self.best_metric,
                'epoch': self.epoch,
        }
        data['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(data, path)

    def load(self, path):
        """
        Load model and other metadata

        Parameters
        ----------
        path : str
            Path of the file to be loaded
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.best_metric = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def log(self, message):
        """
        Log training ouput into console and file

        Parameters
        ----------
        message : str
            Message to be logged
        """
        if self.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


class TorchFitterBoxes(TorchFitter):

    def validation(self, val_loader):
        """
        Validates a model

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            Validation Data
        metric : function with (y_true, y_pred, **metric_kwargs) signature
            Metric to evaluate results on
        metric_kwargs : dict
            Arguments for the passed metric. Ignored if metric is None

        Returns
        -------
        AverageMeter
            object with this epochs's average loss
        float
            calculated metric if a metric is provided, else None
        """
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.verbose:
                if step % self.verbose == 0:
                    print(
                        f'\rVal Step {step}/{len(val_loader)}, '
                        f'summary_loss: {summary_loss.avg:.5f}, '
                        f'time: {(time.time() - t):.5f}', end=''
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
        """
        Run one epoch on the train dataset

        Parameters
        ----------
        train_loader : torch.DataLoader
            DataLoaders containing the training dataset

        Returns
        -------
        AverageMeter
            object with this epochs's average loss
        """
        self.model.train()  # set train mode
        summary_loss = AverageMeter()  # object to track the average loss
        t = time.time()

        # run epoch
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.verbose > 0:
                if step % self.verbose == 0:
                    print(
                        f'\rTrain Step {step}/{len(train_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.5f}', end=''
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
