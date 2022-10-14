import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


def run_training_validation(model, optimizer, data_loaders, n_epochs, verbose=True):
    train_data_loader, validation_data_loader, _ = data_loaders

    metrics = {
        'train_accuracy': [],
        'train_loss': [],
        'val_accuracy': [],
        'val_loss': [],
    }

    for epoch in range(n_epochs):
        model.train()
        train_correct = 0
        train_loss = 0
        for i, data in enumerate(train_data_loader):
            # inputs, labels = data
            inputs = Variable(data[0])
            labels = Variable(data[1])
            optimizer.zero_grad()
            outputs = model(inputs)

            # loss = criterion(outputs, labels)
            loss = F.cross_entropy(outputs, labels)
            train_loss += loss.data
            loss.backward()
            optimizer.step()

            pred = outputs.data.max(1, keepdim=True)[1] 
            train_correct += pred.eq(labels.data.view_as(pred)).long().cpu().sum()

        train_loss /= i+1
        train_acc = 100 * (train_correct / len(train_data_loader.dataset))

        metrics['train_accuracy'].append(train_acc)
        metrics['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0
        val_correct = 0
        
        for i, data in enumerate(validation_data_loader):
            inputs = Variable(data[0])
            labels = Variable(data[1])
            
            outputs = model(inputs)
            
            val_loss += F.cross_entropy(outputs, labels).data

            pred = outputs.data.max(1, keepdim=True)[1] 
            val_correct += pred.eq(labels.data.view_as(pred)).long().cpu().sum()

        val_loss /= i+1
        val_acc = 100 * (val_correct / len(validation_data_loader.dataset))

        metrics['val_accuracy'].append(val_acc)
        metrics['val_loss'].append(val_loss)

        if verbose == 'short':
            # prints epoch progress on single line
            print(f'\rEpoch {epoch + 1}/{n_epochs}', end='' if epoch < n_epochs - 1 else '\n')
        else:
            if verbose:
                print(f'Progress: Epoch {epoch + 1}, Loss: {loss.data}, Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}')
        
    if verbose == 'short':
        best_train_acc = max(metrics['train_accuracy'])
        best_val_acc = max(metrics['val_accuracy'])
        print(f'Best Training Accuracy: {best_train_acc}, Best Validation Accuracy: {best_val_acc}')

    return metrics

def run_test(model, data_loaders, verbose=True):
    _, _, test_data_loader = data_loaders

    model.eval()
    test_loss = 0
    test_correct = 0
    
    for i, data in enumerate(test_data_loader):
        inputs = Variable(data[0])
        labels = Variable(data[1])
        
        outputs = model(inputs)
        
        test_loss += F.cross_entropy(outputs, labels).data

        pred = outputs.data.max(1, keepdim=True)[1] 
        test_correct += pred.eq(labels.data.view_as(pred)).long().cpu().sum()

    # average test_loss
    test_loss /= i+1
    test_acc = 100 * (test_correct / len(test_data_loader.dataset))

    if verbose:
        print(f'Test Loss: {test_loss.data}, Test Accuracy: {test_acc}')

    return test_acc, test_loss