
import torch
from trainer.base import Trainer

class Finetune(Trainer):

    def __init__(self, config):
        super().__init__(config)

    def train_batch(self, x, y, x_=None, y_=None, active_classes=None, task=1, rnt=0.5):
            """ Train model on single batch of current task samples
                and (optionally) replay samples for the Task Incremental
                Learning setting.
                Args:
                    x (torch.Tensor): Input data from current task. 
                    y (torch.LongTensor): Labels from current task.
                    x_ (dict with torch.Tensor): Input data from replay tasks. 
                    y_ (dict with torch.LongTensor): Labels from replay_tasks.
                    active_classes (list): Active classes for each task, (ex: [[0, 1], [2, 3], [4,5], ...])
                    task (int): Task id number starting from 1, e.g. splitMNIST: 1-5
                    rnt (float): Weight constant for current and replay losses (default = 0.5) 
                Returns:
                    loss_dict (dict): Dictionary with loss and accuracy metrics.
            """
            self.model.train()
            self.optimizer.zero_grad()
            # Shorthands
            classes_per_task = self.classes_per_task
            scenario = self.scenario

            # Run model on current task data
            y_hat = self.model(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # -multiclass prediction loss
            predL = None if y is None else self.criterion(y_hat, y) 
            # Weigh losses
            loss_curr = predL
            # Calculate training-precision
            accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # Compute loss
            loss_curr.backward()
            self.optimizer.step()

            # Return the dictionary with different training-loss split in categories
            return {
                'loss_current': loss_curr.item() if x is not None else 0,
                'loss_replay': 0.,
                'accuracy': accuracy if accuracy is not None else 0.,
                'accuracy_replay': 0. ,
            }

    def get_replay_batch(self, task):
        # no replay for finetune
        x_replay = y_replay = None
        return x_replay, y_replay   