# Mamba Stock Prediction

Welcome to our Mamba Stock Prediction project! Here, we harness the power of the Mamba architecture to forecast stock prices accurately. By analyzing historical market data and employing innovative techniques, we demonstrate the effectiveness of Mamba in predicting future stock behavior. Explore our project as we delve into the realm of stock market forecasting with Mamba :)

Video Presentation: https://youtu.be/pZW4XnBDX3M

![Image Description](https://github.com/Eitan-zw/MambaStocks_Project/blob/master/Data/_11520c05-2fec-40dc-9263-98494b359cb9.jpeg)

- [Mamba Stock Prediction](#mamba-stock-prediction)
  * [Data Processing](#data-processing)
  * [Architecture](#architecture)
    + [Model Overview](#model-overview)
    + [Key Components](#key-components)
  * [Requirements](#requirements)
  * [Repository Structure](#repository-structure)
    + [Files](#files)
    + [Data](#data)
  * [Usage](#usage)
  * [Options](#options)
  * [Results](#results)
    + [Example Scenario: Apple Stock](#example-scenario-apple-stock)
    + [Cross-Stock Prediction](#cross-stock-prediction)
  * [References](#references)


## Data Processing

Our data processing begins with the `stockz.py` script, a versatile tool for fetching historical stock data from Yahoo Finance. This script extracts vital metrics like opening and closing prices, high and low prices, adjusted close, and trading volume.

Next, we delve into feature engineering, enriching our dataset with calculated metrics such as daily change, percentage change, and total trading amount. This process enhances our model's understanding of stock price dynamics, bolstering its predictive capabilities.

During model training, we meticulously select the most relevant features while removing any redundant ones. Furthermore, we rigorously split our data into training and testing sets to evaluate the model's performance on unseen data, ensuring its ability to generalize effectively beyond the training phase.

## Architecture

Our stock prediction model is built upon the Mamba framework, specifically leveraging the lightweight implementation available in [mamba-minimal](https://github.com/johnma2006/mamba-minimal). This simplified version encapsulates the essence of Mamba's state space modeling in a single PyTorch file.

![Mamba](https://github.com/state-spaces/mamba/blob/main/assets/selection.png)
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> Paper: https://arxiv.org/abs/2312.00752

### Model Overview

In our implementation, we adapted the Mamba model for stock prediction by integrating it with our custom features and requirements. One notable modification was the removal of the vocabulary embedding layer, aligning the model's architecture with our numerical data processing pipeline.

### Key Components

The Mamba-based model comprises essential components such as residual blocks, selective scanning algorithms, and input-dependent parameter modulation. These elements collectively enable the model to capture intricate patterns in stock price data while ensuring computational efficiency and predictive accuracy.

For further details on the Mamba framework and its implementation, we recommend referring to the original research papers and the annotated code available in the [Mamba](https://github.com/state-spaces/mamba) repository.

## Requirements

The code has been tested running on Python version 3.12, with the following packages and their dependencies installed:
```
numpy==1.26.4
matplotlib==3.8.3
scikit-learn==1.4.1.post1
pandas==2.2.1
pytorch==1.7.1
```
## Repository Structure

### Files

| File Name          | Purpose                                         |
|--------------------|-------------------------------------------------|
| README.md          | Overview of the project and instructions        |
| model.py           | Implementation of the Mamba-based stock prediction model |
| stockz.py          | Script for downloading and processing stock data |
| testing.py         | Script for testing the trained model            |
| training.py        | Script for training the Mamba model             |

### Data

The `data` folder contains the following files:

| File Name                | Purpose                                                  |
|--------------------------|----------------------------------------------------------|
| `aapl_stock_data.csv`    | CSV file containing historical data for Apple stock       |
| `dimensions.csv`         | CSV file containing dimensions used in the model          |
| `nke_stock_data.csv`     | CSV file containing historical data for Nike stock        |
| `trained_model.pth`      | PyTorch model file saved after training                  |


## Usage

stockz.py -> This script is responsible for downloading historical stock data from Yahoo Finance. When you run this script, it will prompt you to enter the ticker symbol of the stock you're interested in, as well as the start and end dates for the historical data. After inputting this information, the script will download the relevant stock data and save it to a CSV file.

training.py -> This script is used to train the stock prediction model on the downloaded historical stock data. Before running this script, make sure you have the CSV file containing the stock data generated by stockz.py. When you run training.py, it will parse the CSV file to load the training data. Additionally, you can customize the training parameters such as the number of epochs, learning rate, and model architecture (hidden layers, dimension of representations, etc.). After training, the script will save the trained model to a file named model.pt.

testing.py -> Once you have a trained model (model.pt) and additional stock data, you can use this script to test the model's performance. Similar to training.py, testing.py requires a CSV file containing stock data. Additionally, you need to provide the path to the trained model (model.pth). When you run testing.py, it will load the trained model and the testing data, and then make predictions based on the testing data. Finally, it will display the predicted stock prices and compare them with the actual prices.

## Options
```
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False, help='Enable CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2, help='Dimension of representations.')
parser.add_argument('--layer', type=int, default=8, help='Number of layers.')
parser.add_argument('--n-test', type=int, default=300, help='Size of test set.')
parser.add_argument('--ts-code', type=str, default='aapl_stock_data', help='Stock code.')
parser.add_argument('--csv-path', type=str, default='dimensions.csv', help='Path to save dimensions.')
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

## Results

### Example Scenario: Apple Stock

In an example scenario with Apple stock, historical data from 1986 to 2024 is downloaded and processed. The model is trained using MSE loss optimization and RMSprop as the optimizer.

Training the model on AAPL stock produced the results shown in Figures 1 and 2. Figure 1 illustrates the AAPL stock prediction result, while Figure 2 displays the loss during training.

![AAPL Stock Prediction Result](https://github.com/Eitan-zw/MambaStocks_Project/blob/master/Data/apple%20stock%20n%3D100.jpg)
                                *Figure 1: AAPL Stock Prediction Result*

![Loss When Training - AAPL](https://github.com/Eitan-zw/MambaStocks_Project/blob/master/Data/epoch%20vs%20loss%2040%20epoch.jpg)
                                *Figure 2: Loss When Training - AAPL*

### Cross-Stock Prediction

To assess the model's generalization capability, we tested the trained model on predicting another stock, NKE (Nike). Figure 3 depicts the prediction results using the AAPL-trained model on NKE stock data.

![NKE Prediction Using AAPL Trained Model](https://github.com/Eitan-zw/MambaStocks_Project/blob/master/Data/nke%20stock%20n%3D100.jpg)
                                *Figure 3: NKE Prediction Using AAPL Trained Model*

## References

1. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces Albert Gu and TriDao](https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf)
2. [Mamba: Redefining Sequence Modeling and Outforming Transformers Architecture](https://www.unite.ai/mamba-redefining-sequence-modeling-and-outforming-transformers-architecture/)
3. [Structured State Spaces: Combining Continuous-Time, Recurrent, and Convolutional Models](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3)
4. [Mamba: The Easy Way](https://jackcook.com/2024/02/23/mamba.html)
5. [A Survey of Forex and Stock Price Prediction Using Deep Learning](https://arxiv.org/ftp/arxiv/papers/2103/2103.09750.pdf)
6. [mamba-minimal](https://github.com/johnma2006/mamba-minimal)
7. [Mamba](https://github.com/state-spaces/mamba)
