'''
 This code is for a data processing and modeling task. It imports necessary libraries,
 loads data, creates a feature extractor, builds a regression model, and makes predictions.
'''

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


# Function to load data
def load_data():
    x_pretrain = pd.read_csv("./jbrodbec/task4/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("./jbrodbec/task4/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("./jbrodbec/task4/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("./jbrodbec/task4/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("./jbrodbec/task4/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

# Neural network class for feature extraction
class Net(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features*0.25)  
        self.fc2 = nn.Linear(in_features * 0.25, in_features*0.125)
        self.fc3 = nn.Linear(in_features*0.125, in_features*0.0625)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(x)
        x = x.squeeze()
        return x

# Function to create a feature extractor
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    model = Net(in_features)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()

    for epoch in tqdm(range(10)):  
        running_loss = 0.0
        model.train()
        print('Epoch %d' % (epoch+1))
        for i in range(0, len(x_tr), batch_size):
            batch_x, batch_y = x_tr[i:i+batch_size], y_tr[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print('Evaluating validation loss' % (epoch+1))
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        print('Epoch %d, train loss: %.6f, val loss: %.6f' % (epoch+1, running_loss/len(x_tr), val_loss))
    
    def make_features(x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float)
            model.eval()
            features = model.fc1(x).numpy()
        return features
    
    return make_features

# Function to create a pretraining class
def make_pretraining_class(feature_extractors):
    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode
        def fit(self, X=None, y=None):
            return self
        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
    
    return PretrainedFeatures

# Function to get a regression model
def get_regression_model():
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    return model

if __name__ == '__main':
    
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    
    # Utilize pretraining data by creating a feature extractor
    feature_extractor = make_feature_extractor(x_pretrain, y_pretrain, batch_size=256, eval_size=1000)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # Regression model
    regression_model = Ridge(alpha=0.1, random_state=42)

    # Set functions for the feature extractor
    loss_function = torch.nn.MSELoss()
    activation = torch.nn.ReLU
    in_features = x_pretrain.shape[1]

    # Define the pipeline
    pipeline = make_pipeline(
        PretrainedFeatureClass(),
        make_column_transformer(
            (StandardScaler(), slice(0, -1)),
            remainder='passthrough'
        ),
        regression_model
    )

    model = Net(in_features)

    # Fit the pipeline on the training data
    pipeline.fit(x_train, y_train)
    
    # Make predictions on the test data
    y_pred = pipeline.predict(x_test)
    
    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")
