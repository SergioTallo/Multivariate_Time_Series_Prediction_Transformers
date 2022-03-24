import numpy as np
import matplotlib.pyplot as plt
from platform import python_version
import pandas as pd
import sklearn
import torch
import seaborn
from tqdm import tqdm


def print_versions():
    print('versions of packages:')
    print(f'Python: {python_version()}')
    print(f'Pandas: {pd.__version__}')
    print(f'Numpy: {np.__version__}')
    print(f'PyTorch: {torch.__version__}')
    print(f'Sklearn: {sklearn.__version__}')
    print(f'seaborn: {seaborn.__version__}')


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


def create_batches(dataset, batch_size, weekdays = False, weekend = False):
  batch_set = []
  pair_set = []

  if weekdays == True:
    counter = 0
    for i in tqdm(range(0, len(dataset) - batch_size)):
      if counter > 901:
        counter += 1
        if counter == 931:
          counter = 0
      else:
        batch_set.append(dataset.iloc[i:i + batch_size, 1:])
        counter += 1

  elif weekend == True:
    counter = 0
    days = 0

    for i in tqdm(range(0, len(dataset) - batch_size)):

      if days < 5:
        
        if counter >= 0 and counter <= 211:
          batch_set.append(dataset.iloc[i:i + batch_size, 1:])
    
        if counter >= 240 and counter <= 480:
          batch_set.append(dataset.iloc[i:i + batch_size, 1:])

        counter += 1

        if counter == 509:
          counter = 0
          days += 1

      else:
        
        if counter >= 0 and counter <= 1411:
          batch_set.append(dataset.iloc[i:i + batch_size, 1:])
        
        counter += 1

        if counter == 1440:
          counter = 0
          days += 1

        if days == 7:
          days = 0

  else:
    for i in tqdm(range(0, len(dataset) - batch_size)):
        batch_set.append(dataset.iloc[i:(i + batch_size), 1:])
    
  for i in range(len(batch_set) - 1):
    data =  batch_set[i]
    target = batch_set[i+1]
    pair_set.append((data, target))

  return np.array(pair_set)
