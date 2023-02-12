import pandas as pd
import numpy as np

import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset




def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    """
    A function created for the purpose of functionaizing Training step in Neural Network.
    Args:
    model (torch.nn.Module) : the target module
    data_loader (torch.utils.data.DataLoader) : the target data loader
    loss_fn (torch.nn.Module) : the target loss function
    optimizer (torch.optim.Optimizer): the target optimizer
    accuracy_fun (function): the target function to calculate accuracy of training and testing phases
    device (torch.device): the target device
    Returns:
    A print out of:
      return True if the function perform successfully
    """
  
    train_loss, train_acc = 0, 0

    #Put Model into training mode
    model.train()

    for batch, (X, y) in enumerate(data_loader):

        #Put Data on target device
        X, y = X.to(device), y.to(device)

        #Forward propagation
        y_pred = model(X)

        #Loss Function (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss

        #Accuaray Function (per batch)
        train_acc += accuracy_fn(y_true=y, 
                               y_pred=y_pred.argmax(dim=1))

        #Optimizer zero grad
        optimizer.zero_grad()

        #Loss backpropagation
        loss.backward()

        #optimizer step
        optimizer.step()
  
    #Calculate the average train loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    #Print out the result of training loop
    print(f'Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f}')
    return True



def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    
    """
    A function created for the purpose of functionaizing Training step in Neural Network.
    Args:
    model (torch.nn.Module) : the target module
    data_loader (torch.utils.data.DataLoader) : the target data loader
    loss_fn (torch.nn.Module) : the target loss function
    accuracy_fun (function): the target function to calculate accuracy of training and testing phases
    device (torch.device): the target device
    Returns:
    A print out of:
      return True if the function perform successfully
    """

    test_loss, test_acc= 0, 0

    #Put the model on testing mode
    model.eval()

    with torch.inference_mode():
        for (test_X, test_y) in (data_loader):

          #Put the data on the same device:
          test_X, test_y = test_X.to(device), test_y.to(device)

          #forward Propagation
          test_pred = model(test_X)

          #Loss Function
          test_loss += loss_fn(test_pred, test_y)

          #Accuacy Function
          test_acc += accuracy_fn(y_true=test_y,
                                 y_pred=test_pred.argmax(dim=1) )

    #Calculate the average test loss / accuracy
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    #Print out the result of testing loop
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")
    return True


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    
    """
    A function created for the purpose of functionaizing Training step in Neural Network.
    Args:
    model (torch.nn.Module) : the target module
    data_loader (torch.utils.data.DataLoader) : the target data loader
    loss_fn (torch.nn.Module) : the target loss function
    accuracy_fun (function): the target function to calculate accuracy of training and testing phases
    device (torch.device): the target device
    Returns:
    A print out of:
      return True if the function perform successfully
    """
    
    loss, acc= 0, 0
    model.eval()
    
    with torch.inference_mode():
        for X, y in data_loader:
            #Put the data on the same device
            X, y= X.to(device), y.to(device)
            #Forward Propagation
            y_pred = model(X)
            # Loss/ Accuracy Function
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y ,y_pred.argmax(dim=1))
    
    # Calculate average loss and accuracy per batch
    loss /= len(data_loader)
    acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}





