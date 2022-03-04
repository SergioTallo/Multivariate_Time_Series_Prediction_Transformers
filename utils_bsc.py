import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from platform import python_version
import pandas as pd
import sklearn
import torch


def print_versions():
    print('versions of packages:')
    print(f'Python: {python_version()}')
    print(f'Pandas: {pd.__version__}')
    print(f'Numpy: {np.__version__}')
    print(f'PyTorch: {torch.__version__}')
    print(f'Sklearn: {sklearn.__version__}')


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


def create_batches(dataset, size, device):
    dailyset = []
    j = 0

    for i in range(1440, dataset.shape[0] + 1, 1440):
        temp_set = dataset.iloc[j:i]

        j = i

        dailyset.append(temp_set)

    weekdayset = []

    for i in range(7):
        for j in range(0, 5):
            if (j + 7 * i) < len(dailyset):
                weekdayset.append(dailyset[j + 7 * i])
            else:
                break

    end_set = []

    for n in weekdayset:
        temp_set = n.iloc[240:1170]
        end_set.append(temp_set)

    batch_set = []

    for m in end_set:
        for i in range(0, len(m) - size):
            data_set = m.iloc[i:i + size, 1:]
            if device != 'cuda:0':
                print('CPU')
                torch_tensor = torch.tensor(data_set.values)
                print(torch_tensor.get_device())
                batch_set.append(torch_tensor)
            else:
                print('GPU')
                torch_tensor = torch.cuda.tensor(data_set.values)
                print(torch_tensor.get_device())
                batch_set.append(torch_tensor)

    return batch_set
