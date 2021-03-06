import torch
import gc
import os
import logging
import torch.optim as optim
import torch.nn as nn
from datetime import datetime

"""
5 Fold CV Best Loss: 0.6709386929869652 | 0.6842632591724396 | 0.6594144776463509 | 0.6695199608802795 | 0.6728682070970535
"""

# ==============================================
# Setup
# ==============================================

# Model / Data Set Choice
from rick.experiments.models.exp2_dynamic_image_model import get_model
from rick.experiments.datasets.exp2_dynamic_image_dataset import get_training_and_validation_dataloaders
from rick.experiments.utilities.calculate_metrics import calculate_accuracy

# GPU
device = torch.device("cuda:0")

# Training Settings
batch_size = 128
num_epochs = 50
folds_to_train = [0, 1, 2, 3, 4]

# Load data-loaders, i.e. [(train_dataloader, val_dataloader), ...]
data_loaders = get_training_and_validation_dataloaders(n_splits=5, batch_size=batch_size)

# File-name
file_name = ''.join(os.path.basename(__file__).split('.py')[:-1])

# Logging
logger = logging.getLogger(file_name)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('{}.log'.format(file_name))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


# ==============================================
# Train K-Folds
# ==============================================

best_loss_per_fold = {}
for fold_i, (train_loader, validation_loader) in enumerate(data_loaders):
    if fold_i not in folds_to_train:
        print('Skipping Fold: {}'.format(fold_i))
        continue

    # Get Model
    net = get_model()
    net.to(device)

    # Initialize optimizer and loss function.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    time_start = datetime.now()
    print('Starting Training on Fold: {}\n'.format(fold_i))

    best_val_loss = float('inf')
    for epoch_i in range(num_epochs):
        train_running_loss = 0.0
        train_running_acc = 0.0
        num_mini_batches = 0

        # ==============================================
        # Training Pass
        # ==============================================
        net.train()
        for i, data in enumerate(train_loader):
            x_train, y_train = data
            x_train, y_train = x_train.to(device), y_train.to(device)

            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Prediction.
            y_pred = net(x_train)

            # Calculate Loss.
            loss = criterion(y_pred, y_train)

            # Step
            loss.backward()
            optimizer.step()

            # Keep track of the loss and number of batches.
            num_mini_batches += 1
            train_running_loss += loss.item()
            train_running_acc += calculate_accuracy(y_pred, y_train)

        # ==============================================
        # Validation Pass
        # ==============================================
        val_running_loss = 0.0
        val_running_acc = 0.0
        num_val_mini_batches = 0

        net.eval()
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                x_val, y_val = data
                x_val, y_val = x_val.to(device), y_val.to(device)

                # Prediction.
                y_pred = net(x_val)

                # Calculate Loss.
                loss = criterion(y_pred, y_val)

                # Keep track of the loss and number of batches.
                num_val_mini_batches += 1
                val_running_loss += loss.item()
                val_running_acc += calculate_accuracy(y_pred, y_val)

        # ==============================================
        # Statistics
        # ==============================================
        time_elapsed = datetime.now() - time_start
        avg_loss = train_running_loss / num_mini_batches
        avg_acc = train_running_acc / num_mini_batches

        avg_val_loss = val_running_loss / num_val_mini_batches
        avg_val_acc = val_running_acc / num_val_mini_batches

        # Keep track of best model.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Output training status.
        output_msg = 'Epoch: {}/{}\n' \
                     '---------------------\n' \
                     'train loss: {:.6f}, val loss: {:.6f}\n' \
                     'train acc: {:.4f}, val acc: {:.4f}\n' \
                     'best val loss: {:.6f}, time elapsed: {}\n'. \
            format(epoch_i + 1, num_epochs,
                   avg_loss, avg_val_loss,
                   avg_acc, avg_val_acc,
                   best_val_loss, str(time_elapsed).split('.')[0])
        print(output_msg)
        logger.info(output_msg)

    best_loss_per_fold[fold_i] = best_val_loss
    del net
    gc.collect()


print('Finished Training')
print("Best Loss per Fold:\n", best_loss_per_fold)
