import numpy as np
import torch
import time
import copy
import os


class NN_Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(),
                 scheduler=False, milestones=[10], gamma=0.1):
        # if the optimizer is Adam, the default optimization parameters are updated only.
        if optim == torch.optim.Adam:
            optim_args_merged = self.default_adam_args.copy()
            optim_args_merged.update(optim_args)
        else:
            optim_args_merged = optim_args

        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.milestones = milestones
        self.gamma = gamma

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, saving_path=None):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        # set up the optimizer
        optimizer = self.optim(model.parameters(), **self.optim_args)

        # set upt the scheduler if any:
        if self.scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        
        self._reset_histories()
        ################################
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        ################################

        print('START TRAINING')

        since = time.time()

        # for always saving the state dictionary of the actual best model.
        best_model_state_dict = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0

        iteration_counter = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for param_group in optimizer.param_groups:
                actual_learning_rate = param_group['lr']
            print('learning rate : {}'.format(actual_learning_rate))
            print('-' * 10)

            # training and validation phases:
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode

                    iter_in_epoch = 0

                    epoch_loss = 0
                    epoch_acc = 0
                    data_size = 0
                    for batch in train_loader:
                        inputs, labels = batch
                        iter_in_epoch += 1
                        ################################
                        #inputs = inputs.to(device)
                        #labels = labels.to(device)
                        ################################

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward:
                        outputs = model(inputs)

                        # loss calculation:
                        # print('output: {}'.format(outputs))
                        # print('lables: {}'.format(labels))
                        loss = self.loss_func(outputs, labels)

                        # back propagation:
                        loss.backward()
                        optimizer.step()

                        # logging the loss and the accuracy:
                        batch_data_size = labels.shape[0]
                        pred_labels = outputs.argmax(dim=1)
                        accuracy = np.mean((pred_labels == labels).cpu().numpy())

                        epoch_loss += loss.item()
                        epoch_acc += accuracy * batch_data_size     # for weighing average of the accuracy
                        data_size += batch_data_size

                        # printing the accuracy in every log_nth iteration:
                        if log_nth != 0:
                            if (iteration_counter % log_nth) == 0:
                                print('Iteration {}/{} train accuracy: {}, train_loss: {}'
                                    .format(iteration_counter, len(train_loader), accuracy, loss.item()))                           

                        iteration_counter += 1

                    # saving the training loss and accuracy in every epoch:
                    self.train_loss_history.append(epoch_loss / data_size)
                    self.train_acc_history.append(epoch_acc / data_size)

                else:
                    model.eval()  # Set model to evaluate mode

                    epoch_loss, epoch_acc = 0, 0
                    data_size = 0
                    for batch in val_loader:
                        inputs, labels = batch
                        #############################
                        #inputs = inputs.to(device)
                        #labels = labels.to(device)
                        #############################
                        # forward pass:
                        outputs = model(inputs)

                        # loss calculation:
                        loss = self.loss_func(outputs, labels)

                        # logging the loss and the accuracy:
                        batch_data_size = labels.shape[0]
                        pred_labels = outputs.argmax(dim=1)
                        accuracy = np.mean((pred_labels == labels).cpu().numpy())

                        epoch_loss += loss.item()
                        epoch_acc += accuracy * batch_data_size     # for weighing average of the accuracy
                        data_size += batch_data_size

                    self.val_loss_history.append(epoch_loss / data_size)
                    self.val_acc_history.append(epoch_acc / data_size)

            if self.scheduler:
                lr_scheduler.step()

            # printing the loss and accuracy info for the epoch:
            print('EPOCH {}/{} TRAIN loss/acc : {:.3f}/{:.2f}%'
                    .format(epoch, num_epochs-1, self.train_loss_history[-1]*1000, self.train_acc_history[-1] * 100))
            print('EPOCH {}/{} VAL loss/acc : {:.3f}/{:.2f}%'
                    .format(epoch, num_epochs - 1, self.val_loss_history[-1]*1000, self.val_acc_history[-1] * 100))
            print('-' * 10)

            # storing and saveing the best model:
            if self.val_acc_history[-1] >= best_val_acc:
                best_model_state_dict = copy.deepcopy(model.state_dict())
                # saving the best model's state dictionary:
                if saving_path is not None:
                    try: 
                        torch.save(model.state_dict(), saving_path)
                    except:
                        print('unable to save model to ' + saving_path)

                best_val_acc = self.val_acc_history[-1]

        print('FINISH.')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        torch.cuda.empty_cache()

        # loading the best acquired state dictionary to the model:
        model.load_state_dict(best_model_state_dict)