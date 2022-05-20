import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm


def week_plot(data, col, name):
    days = [0, 1440, 2880, 4320, 5760, 7200, 8640]
    daysname = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    dataret = []

    j = 0
    count = 1

    for i in range(10080, data.shape[0], 10080):

        dataplot = np.array(data.iloc[j:i, col])

        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(0, dataplot.shape[0]), dataplot, label=name)

        for k in range(len(days)):
            plt.axvline(days[k], color='gray')
            plt.text(days[k] - 5, max(dataplot), daysname[k])

        plt.legend()
        plt.title(f'{name} week: {count}')
        plt.xlabel('Sample')
        plt.ylabel('y')
        plt.show()
        j = i
        count += 1

        dataret.append(dataplot)

    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.set_title(name)
    ax1.boxplot(dataret)
    plt.show()


def daily_plot(data, col, name):
    j = 0
    dataday = []
    count = 1

    for i in range(1440, data.shape[0], 1440):
        PLN_1 = np.array(data.iloc[j + 240:j + 1170, col])

        j = i

        dataday.append(PLN_1)

    datafix = []

    for i in range(6):
        for j in range(0, 5):
            datafix.append(dataday[j + 7 * i])

    for k in datafix:

        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(0, len(k)), k, label=name)

        for n in range(0, len(k + 1), 60):
            plt.axvline(n, color='gray')

        plt.legend()
        plt.title(f'{name} day: {count}')
        plt.xlabel('Sample')
        plt.ylabel('y')
        plt.show()
        count += 1

    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.set_title(f'{name} daily')
    ax1.boxplot(datafix)
    plt.show()

# apply the mean / stddev scaling in Pandas using the .mean() and .std() methods
def normalize_mean_std_dataset(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply mean / stddev scaling
    for column in tqdm(df_norm.columns):
        if column != 'time':
            df_norm[column] = (df_norm[column] - df_norm[column].mean()) / df_norm[column].std()
    return df_norm


def create_sequece_dataloaders(dataset, seq_length, batch_size, device):

    # Create a dataset with pairs data / next /Target (in this case data is one
    # sequence of seq_length measures (18 features), next is the next value in the sequence
    # and target is the following value with the
    # measurements (18 features)). When you plug in one measurement, the model should out the next measurement

    assert seq_length > 1, f"sequence length should be greater than 1 expected, got: {seq_length}"

    pair_set = []

    print('Creating train/test data loaders')

    for i in tqdm(range(len(dataset) - (seq_length +1))):
        data = np.array(dataset.iloc[i:i+seq_length, 1:])
        next = np.array(dataset.iloc[i+seq_length, 1:], dtype= float)
        target = np.array(dataset.iloc[i+(seq_length + 1), 1:], dtype= float)

        pair_set.append((data, next, target))

    dataset_pairs = np.array(pair_set, dtype=object)

    training_data_pairs, testing_data_pairs = train_test_split(dataset_pairs, test_size=0.1)

    data = []
    next = []
    target = []

    for i in training_data_pairs:
        data.append(i[0])
        next.append(i[1])
        target.append(i[2])

    training_data = torch.from_numpy(np.array(data)).float().to(device)
    training_next = torch.from_numpy(np.array(next)).float().to(device)
    training_target = torch.from_numpy(np.array(target)).float().to(device)

    data = []
    next = []
    target = []

    for i in testing_data_pairs:
        data.append(i[0])
        next.append(i[1])
        target.append(i[2])

    test_data = torch.from_numpy(np.array(data)).float().to(device)
    test_next = torch.from_numpy(np.array(next)).float().to(device)
    test_target = torch.from_numpy(np.array(target)).float().to(device)

    print(f'Sequence length: {seq_length}')
    print(f'Batch size: {batch_size}')
    print(f'length of training set: {training_data.shape[0]}')
    print(f'length of test set: {test_data.shape[0]}')
    print('\n')

    # Create data loader to feed the model in mini batches
    loader_train = DataLoader(
        dataset=torch.utils.data.TensorDataset(training_data, training_next, training_target),
        batch_size=batch_size,
        shuffle=True
    )

    # Create data loader for testing the model
    loader_test = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(test_data, test_next, test_target),
        batch_size=batch_size,
        shuffle=True
    )

    return loader_train, loader_test


def create_dataloaders(dataset, device):

    # Create a dataset with pairs data / Target (in this case data is one measure (18 features) and target is the next measure (18 features))
    # When you plug in one measure, the model should out the next measure

    pair_set = []

    for i in tqdm(range(len(dataset) -1)):
        data = np.array([j for j in dataset.iloc[i, 1:]])
        target = np.array([j for j in dataset.iloc[i+1, 1:]])

        pair_set.append((data, target))

    dataset_pairs = np.array(pair_set)

    training_data_pairs, testing_data_pairs = train_test_split(dataset_pairs, test_size=0.1)

    data = []
    target = []
    for i in training_data_pairs:
        data.append(i[0])
        target.append(i[1])

    training_data = torch.from_numpy(np.array(data)).float().to(device)
    training_target = torch.from_numpy(np.array(target)).float().to(device)

    data = []
    target = []
    for i in testing_data_pairs:
        data.append(i[0])
        target.append(i[1])

    test_data = torch.from_numpy(np.array(data)).float().to(device)
    test_target = torch.from_numpy(np.array(target)).float().to(device)

    print(f'length of training set: {training_data.shape[0]}')
    print(f'length of test set: {test_data.shape[0]}')
    print('\n')

    # Create data loader to feed the FFN in mini batches

    loader_train = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(training_data, training_target),
        batch_size=16,
        shuffle=True
    )

    # Create data loader for testing the model
    loader_test = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(test_data, test_target),
        batch_size=16,
        shuffle=True
    )

    return loader_train, loader_test

class ANN_relu(nn.Module):

    def __init__(self, D_in, D_out):
        super(ANN_relu, self).__init__()
        self.linear1 = nn.Linear(D_in, 180)
        self.linear2 = nn.Linear(180, 640)
        self.linear3 = nn.Linear(640, 180)
        self.linear4 = nn.Linear(180, D_out)

        self.relu = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.linear4(x)

# This function trains the model for one epoch
def train_FFN(model, criterion, optimizer, train_loader, test_loader, n_epochs, train_loss = None, test_loss = None):

    if train_loss is not None:
        epoch_loss_train = train_loss
        best_train_loss = min([np.mean(i) for i in train_loss])
        best_epoch = np.where(min([np.mean(i) for i in test_loss]))
    else:
        epoch_loss_train = []
        best_train_loss = 9999999999
        best_epoch = 0

    if test_loss is not None:
        epoch_loss_test = test_loss
        best_test_loss = min([np.mean(i) for i in test_loss])
        best_epoch = np.where(min([np.mean(i) for i in train_loss]))
    else:
        epoch_loss_test = []
        best_test_loss = 99999999999
        best_epoch = 0

    best_model = model
    starting_epoch = len(epoch_loss_test)

    for e in range(1, n_epochs +1):
        print(f'\nEpoch {e + starting_epoch}:')

        print('Train')
        model.train()

        for i in tqdm(train_loader):

            data, target = i[0], i[1]

            optimizer.zero_grad()

            # Forward Pass
            output = model(data)

            #Compute loss
            loss = criterion(output, target)

            #Backpropagation
            loss.backward()

            #Optimization
            optimizer.step()

        losses_train = []

        print('\nTest with training set')
        model.eval()
        with torch.no_grad():
            for i in tqdm(train_loader):

                data, target = i[0], i[1]

                output = model(data)

                losses_train.append (float(criterion(output, target).item()))

        print('\nCurrent Mean loss Train: ', np.mean(losses_train))
        epoch_loss_train.append(losses_train)

        losses_test = []

        print('\nTest with test set')
        model.eval()
        with torch.no_grad():
            for i in tqdm(test_loader):

                data, target = i[0], i[1]

                output = model(data)

                losses_test.append (float(criterion(output, target).item()))


        print('\nCurrent Mean loss: ', np.mean(losses_test))
        epoch_loss_test.append(losses_test)

        if np.mean(losses_test) < best_test_loss:
            best_test_loss = np.mean(losses_test)
            best_train_loss = np.mean(losses_train)
            best_model = model
            best_epoch = e

    return (best_model, best_train_loss, best_test_loss, best_epoch), epoch_loss_train, epoch_loss_test

def print_results_training(train_loss, test_loss, title, test_loss_baseline = None, baseline_label = None):

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)
    plt.ion()

    fig.show()
    fig.canvas.draw()

    if test_loss_baseline is not None:
        ax.plot(test_loss_baseline, label=baseline_label)
    ax.plot([np.mean(i) for i in train_loss], label= 'Train_loss')
    ax.plot([np.mean(i) for i in test_loss], label= 'Test_loss')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    fig.canvas.draw()

def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> int:
    """
    Count (trainable) paramaters of specified model.

    :param model: model for which to compute the amount of (trainable) parameters
    :param only_trainable: only include trainable parameters in total count
    :return: amount of (trainable) parameters of the specified model
    """

    return sum(parameter.numel() for parameter in model.parameters() if any(
        (not only_trainable, only_trainable and parameter.requires_grad)))

class Transformer(nn.Module):
    def __init__(self, feature_size, output_size, num_encoder_layers, num_heads, num_decoder_layers, device, dim_feedforward: int=2048, dropout: float =0.1, batch_first: bool = False):
        super(Transformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model= feature_size, nhead= num_heads, dim_feedforward=dim_feedforward, dropout=dropout, device=device, batch_first=batch_first)
        decoder_layer = nn.TransformerDecoderLayer(d_model= feature_size, nhead= num_heads, dim_feedforward=dim_feedforward, dropout=dropout, device=device, batch_first=batch_first)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers= num_decoder_layers)

        self.output_size = output_size
        self.device = device

    def generate_square_mask(self, dim):
        return torch.triu(torch.ones(dim, dim) * float('-inf'), diagonal=1).to(self.device)

    def positional_encoding(self, seq_len: int, dim_model: int, device):

        position_encoding = torch.zeros(seq_len, dim_model)

        for pos in range(seq_len):
            for i in range(0, int(dim_model / 2)):
                position_encoding[pos, 2 * i] = math.sin(pos / (10000 ** ((2 * i)/dim_model)))
                position_encoding[pos, (2 * i) + 1] = math.cos(pos / (10000 ** ((2 * i)/dim_model)))

        position_encoding = position_encoding.to(device)

        return position_encoding

    def forward (self, enc_input, dec_input):

        memory_mask = self.generate_square_mask(len(enc_input))

        src_pos_enc = enc_input + self.positional_encoding(seq_len= enc_input.shape[1], dim_model= enc_input.shape[2], device= self.device)
        src_pos_dec = dec_input + self.positional_encoding(seq_len= dec_input.shape[1], dim_model= dec_input.shape[2], device= self.device)

        output = self.encoder (src= src_pos_enc, mask=None)
        output = self.decoder (tgt= src_pos_dec, memory= output, tgt_mask=None, memory_mask=None)

        return output

def training_transformer(model, optimizer, criterion, train_loader, test_loader, n_epochs, train_loss = None, test_loss = None):

    if train_loss is not None:
        epoch_loss_train = train_loss
        best_train_loss = min([np.mean(i) for i in train_loss])
        best_epoch = np.where(min([np.mean(i) for i in test_loss]))
    else:
        epoch_loss_train = []
        best_train_loss = 9999999999
        best_epoch = 0

    if test_loss is not None:
        epoch_loss_test = test_loss
        best_test_loss = min([np.mean(i) for i in test_loss])
        best_epoch = np.where(min([np.mean(i) for i in train_loss]))
    else:
        epoch_loss_test = []
        best_test_loss = 99999999999
        best_epoch = 0

    best_model = model
    starting_epoch = len(epoch_loss_test)

    for e in range(1, n_epochs + 1):

        print(f'Epoch: {e + starting_epoch} of {n_epochs}')
        print('Training...')
        model.train()

        for i in tqdm(train_loader):

            input = i[0]
            out = i[1].unsqueeze(0).permute(1,0,2)
            target = i[2].unsqueeze(0).permute(1,0,2)

            net_out = model.forward(input, out)

            #Compute loss
            loss = criterion(net_out, target)

            optimizer.zero_grad()

            #Backpropagation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            #Optimization
            optimizer.step()


        print('\nTest with training set')
        losses_train = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(train_loader):

                input = i[0]
                out = i[1].unsqueeze(0).permute(1,0,2)
                target = i[2].unsqueeze(0).permute(1,0,2)

                net_out = model.forward(input, out)

                #Compute loss
                losses_train.append (float(criterion(net_out, target).item()))


        print('\nCurrent Mean loss Train Set: ', np.mean(losses_train))
        epoch_loss_train.append(losses_train)

        print('\nTest with test set')
        losses_test = []
        model.eval()


        with torch.no_grad():
            for i in tqdm(test_loader):

                input = i[0]
                out = i[1].unsqueeze(0).permute(1,0,2)
                target = i[2].unsqueeze(0).permute(1,0,2)

                net_out = model.forward(input, out)

                #Compute loss
                losses_test.append (float(criterion(net_out, target).item()))

        print('\nCurrent Mean loss Test Set: ', np.mean(losses_test))
        epoch_loss_test.append(losses_test)

        print('\n')

        if np.mean(losses_test) < best_test_loss:
            best_test_loss = np.mean(losses_test)
            best_train_loss = np.mean(losses_train)
            best_model = model
            best_epoch = e

    return (best_model, best_train_loss, best_test_loss, best_epoch), epoch_loss_train, epoch_loss_test

def define_train_transformers(models, device, dataset, training_results_transformers, path_save, colab):
    for i in models:
        if models[i][7] is True:

            loader_train, loader_test = create_sequece_dataloaders(dataset=dataset, seq_length=models[i][8], batch_size=models[i][9], device=device)

            # Initialize Transformer Model and Optimizer
            model = Transformer (num_encoder_layers=models[i][0],
                                           num_decoder_layers=models[i][1],
                                           feature_size=18,
                                           output_size=18,
                                           num_heads=models[i][2],
                                           dim_feedforward=models[i][3],
                                           device = device,
                                           batch_first=True)

            print(f'Model: {type(model).__name__} - {i}')
            print(f'{count_parameters(model)} trainable parameters.')

            n_epochs = 200
            learning_rate = 0.01

            if models[i][4] == 'SGD':

                if models[i][6] is not None:
                    optimizer = torch.optim.SGD(model.parameters(), lr=models[i][5], momentum=models[i][6])
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=models[i][5])
            elif models[i][4] == 'ADAM':
                optimizer = torch.optim.Adam(model.parameters(), lr=models[i][5])

            criterion = nn.MSELoss()

            start_time = datetime.now()

            best_results, train_losses, test_losses = training_transformer(
                model= model,
                optimizer= optimizer,
                criterion= criterion,
                train_loader= loader_train,
                test_loader= loader_test,
                n_epochs= n_epochs)


            Transformer_trained_Model = best_results[0]
            best_train_loss = best_results[1]
            best_test_loss = best_results[2]
            best_epoch_number = best_results[3]

            end_time = datetime.now()
            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds()

            print(f'Best test loss at epoch {best_epoch_number}')
            print(f'Train Loss: {best_train_loss}')
            print(f'Test Loss: {best_test_loss}')
            print(f'\nTraining time for {n_epochs} epochs: {execution_time} seconds')

            print(f'Training time: {execution_time} seconds')

            training_results_transformers[i] = [Transformer_trained_Model, best_train_loss, best_test_loss, best_epoch_number, train_losses, test_losses, execution_time]

            # save to npy file
            np.save(path_save + '/Transformer_' + i + '_train.npy', train_losses)
            np.save(path_save + '/Transformer_' + i + '_test.npy', test_losses)
            torch.save(Transformer_trained_Model, path_save + '/Transformer_' + models[i][8] + '.pt')

            if colab is True:
                from google.colab import files

                files.download(path_save + '/Transformer_' + i + '_train.npy')
                files.download(path_save + '/Transformer_' + i + '_test.npy')
                files.download(path_save + '/Transformer_' + i + '.pt')

    return training_results_transformers