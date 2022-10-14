import enum
from statistics import mode
import matplotlib.pyplot as plt

# models: CNN, RCNN w/ LSTM, RCNN w/ GRU


# Plot Results for Single Subject(s)
#   plots the the best accuracies obtained for each individual subject
#   *** not sure if its necessary to plot this or if we're even gonna do this -- maybe just a table ***
# 
# INPUTS: 
#   best accuracy (train, test, val) for each subject
#       e.g. train is an array of max len 9 (one [best] accuracy for each subject)
#   model name: choose from ['CNN', 'RCNN w/ LSTM', 'RCNN w/ GRU'] (default: 'CNN')
def plot_single_subjects(test_acc, model='CNN'):
    n_subjects = len(test_acc)
    if n_subjects > 9:
        print('error: too many subjects for best accuracies -- max 9')
        return

    # plt.figure()
    # for i,data in enumerate([train, test, val]):
    #     labels = ['Train', 'Test', 'Validation']
    #     if data:
    #         if len(data) != n_subjects:
    #             print(f'error: inconsistent array size for {labels[i]} Accuracies')
    #             return
    #         plt.scatter(range(n_subjects), data, label=labels[i])
            
    # plt.title(f'Performance of {model} Model for Individual Subjects')
    # plt.xlabel('Subject ID')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # if show:
    #     plt.show()

    plt.figure()
    plt.scatter(range(n_subjects), test_acc)
    plt.title(f'Performance of {model} Model for Individual Subjects')
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')

    plt.show()


###   Lea's version ###
def plot_loss_acc(data_loss_train, data_loss_val, data_acc_train, data_acc_val, model='CNN'):
    n_epochs = len(data_loss_train) #metrics['train_loss']

    #Unpack tensors
    train_loss = [t.item() for t in data_loss_train]
    val_loss = [t.item() for t in data_loss_val]

    train_acc = [t.item() for t in data_acc_train]
    val_acc = [t.item() for t in data_acc_val]

    #Plot both
    plt.figure(figsize=(10,7))

    plt.subplot(221)
    plt.plot(range(n_epochs), train_loss, label='Training')
    plt.plot(range(n_epochs), val_loss, label='Validation')
    plt.title(f'Loss of {model}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(222)
    plt.plot(range(n_epochs), train_acc, label='Training')
    plt.plot(range(n_epochs), val_acc,  label='Validation')
    plt.title(f'Accuracy of {model}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# Plot Results for All Subjects
#   plots epochs vs. accuracies or loss for [train, test, val]
# 
# INPUTS: 
#   arrays of accuracies by epoch for [train, test, val]
#   model name: choose from ['CNN', 'RCNN w/ LSTM', 'RCNN w/ GRU'] (default: 'CNN')
#   evaluation_metric: default 'Accuracy', but can be changed to 'Loss'
def plot_single_model(train, test=None, val=None, model='CNN', evaluation_metric='Accuracy', show=True):
    n_epochs = len(train)

    plt.figure()
    for i,data in enumerate([train, test, val]):
        labels = ['Train', 'Test', 'Validation']
        data = data.item()
        if data:
            if len(data) != n_epochs:
                print(f'error: inconsistent array size for {labels[i]}')
                return
            plt.plot(range(n_epochs), data, label=labels[i])
            
    plt.title(f'{evaluation_metric} of {model} Model over All Subjects')
    plt.xlabel('Epoch')
    plt.ylabel('{evaluation_metric}')
    plt.legend()

    if show:
        plt.show()


# Plot Comparison of Accuracies Between Models
#   plots epochs vs. accuracies for specified accuracy type (train, test, val) ...
#       for each model data provided ('CNN', 'RCNN w/ LSTM', 'RCNN w/ GRU')
# 
# INPUTS: 
#   accuracy data by epoch for each model specified (CNN, LSTM, GRU)
#   accuracy type - which accuracy data is being compared (train, test, val)
#   evaluation_metric: default 'Accuracy', but can be changed to 'Loss'
def plot_compare_models(cnn=None, lstm=None, gru=None, acc_type='val', evaluation_metric='Accuracy', show=True):
    acc_names = {
        'train': 'Training',
        'test': 'Testing',
        'val': 'Validation', 
        'validation': 'Validation',
    }
    n_epochs = len(cnn)

    plt.figure()
    models = []
    for i,data in enumerate([cnn, lstm, gru]):
        labels = ['CNN', 'RCNN w/ LSTM', 'RCNN w/ GRU']
        short_labels = ['CNN', 'LSTM', 'GRU']

        if data:
            if len(data) != n_epochs:
                print(f'error: inconsistent array size for {labels[i]}')
                return
            plt.plot(range(n_epochs), data, label=short_labels[i])
            models.append(labels[i])
            
    models = ', '.join(models)
    plt.title(f'Comparing {acc_names[acc_type.lower()]} {evaluation_metric} of [{models}] Models over All Subjects')
    plt.xlabel('Epoch')
    plt.ylabel('{evaluation_metric}')
    plt.legend()

    if show:
        plt.show()
        


# all subjects as function of time
#   *** compare models or plot each one individually? *** 
def plot_all_subjects_over_time(time_data, cnn_acc, lstm_acc, gru_acc, show=True):
    plt.figure()

    plt.plot(time_data, cnn_acc, '-o', label='CNN')
    plt.plot(time_data, lstm_acc, '-o', label='CNN+LSTM')
    plt.plot(time_data, gru_acc, '-o', label='CNN+GRU')
    
    plt.title('EEG Signal Duration vs Model Test Accuracy')
    plt.xlabel('Time (s)')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()

    if show:
        plt.show()
