import torch
import torch.nn as nn
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def train(data_loader, model, optimizer, device, batch_size):
    model.train()
    count = 0
    h = model.init_hidden(batch_size, device, model)
    for data in data_loader:
        # print(data)
        # print(count)
        h = tuple([e.data for e in h])
        reviews = data['review']
        targets = data['target']
        # print(reviews.size())
        reviews = reviews.to(device, dtype = torch.long)
        targets = targets.to(device, dtype = torch.float)
        
        predictions, h = model(reviews, h)
        # predictions = model(reviews)
        # print('Predictions and targets check', predictions.size(), targets.view(-1,1).size())
        loss = nn.BCEWithLogitsLoss()
        output = loss(predictions, targets.view(-1,1))
        optimizer.zero_grad()
        # print('Accuracy: ', binary_acc(predictions, targets.view(-1,1)))
        count = count + 1
        output.backward()
        optimizer.step()

def evaluate(data_loader, model, device, batch_size, optimizer, scheduler):
    final_predictions = []
    final_targets = []

    h = model.init_hidden(batch_size, device, model)
    model.eval()

    with torch.no_grad():
        for data in data_loader:

            h = tuple([e.data for e in h])
            reviews = data['review']
            targets = data['target']
            reviews = reviews.to(device, dtype = torch.long)
            targets = targets.to(device, dtype = torch.float)

            predictions, h = model(reviews, h)
            loss = nn.BCEWithLogitsLoss()
            valid_step_loss = loss(predictions, targets.view(-1,1))
            valid_loss = valid_step_loss.item() * reviews.size(0)
            # predictions = model(reviews)
            # print('Val Accuracy: ', binary_acc(predictions, targets.view(-1,1)))
            # print('Targets raw', targets.size(), targets)
            # print('Prediction Raw', predictions.size(), predictions)
            predictions = predictions.cpu().numpy().tolist()
            # print('Predictions after sigmoid', sigmoid(np.array(predictions)))
            targets = data['target'].cpu().numpy().tolist()
          
            final_predictions.extend(predictions)
            final_targets.extend(targets)
    return final_predictions, final_targets, valid_loss