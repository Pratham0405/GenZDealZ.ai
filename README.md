# Predicting User Purchase History with Deep Learning

## Overview
This project aims to predict the next purchase a user might make based on their previous purchase history. The model developed will enhance the recommendation system of GenZDealZ.ai by providing personalized recommendations.

## Repository Structure
- `data_simulation.py`: Script to generate simulated purchase history data.
- `model.py`: Script to preprocess data, build, and train the deep learning model.
- `evaluate.py`: Script to evaluate the trained model's performance.
- `requirements.txt`: List of dependencies required to run the scripts.

## Getting Started

### Prerequisites
Ensure you have Python installed (version 3.6 or higher). Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Scripts
### Generate Simulated Data:
Run the `data_simulation.py` script in terminal to generate the purchase history data.

```bash
python data_simulation.py
```
This script will create a file named simulated_purchase_history.json containing the simulated data.

### Build and Train the Model:
Run the `model.py` script to preprocess the data, build the model, and train it.

```bash
python model.py
```
This script will load the simulated_purchase_history.json file, preprocess the data, and train an RNN model. The trained model will be saved as model.h5.

### Evaluate the Model:
Run the `evaluate.py` script in terminal to evaluate the performance of the trained model.

```bash
python evaluate.py
```
This script will load the trained model (model.h5) and evaluate it on the test data, printing the accuracy and loss metrics.


## Thankyou
