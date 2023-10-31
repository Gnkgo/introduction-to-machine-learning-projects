'''
This code performs a binary classification task using feature embeddings
and deep learning. It generates embeddings from images, trains a neural network,
and tests the model for binary predictions.
'''

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to generate feature embeddings from images
def generate_embeddings():
    # Define a transform to pre-process the images
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    # Use ResNet50 as the pre-trained model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Set the model to evaluation mode (important for batch normalization, dropout, etc.)
    model.eval()

    # Move the model to the device (CPU or GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load the dataset with the defined transform
    train_dataset = datasets.ImageFolder(root="./dataset/", transform=preprocess)

    # Set up the data loader for the dataset
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    # Get the number of images in the dataset and initialize the embeddings array
    num_images = len(train_dataset)
    embedding_size = list(model.children())[-1].in_features
    embeddings = np.zeros(num_images, embedding_size)
      
    # Remove the last layer of the model
    model = nn.Sequential(*list(model.children())[:-1]) 

    # Extract embeddings for each image in the dataset
    with torch.no_grad():  
        for i, (image, _) in enumerate(tqdm(train_loader)): 
            # Get the features for the image using the pre-trained model
            features = model(image).squeeze().cpu().numpy()

            # Calculate the start and end indices for the current batch of images
            batch_start = i * train_loader.batch_size
            batch_end = (i + 1) * train_loader.batch_size

            # Add the features to the embeddings array
            embeddings[batch_start:batch_end] = features

    # Normalize the embeddings
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)              

    # Save the embeddings to a file
    np.save('./jbrodbec/task3/embeddings.npy', embeddings)

# Function to get data from triplets
def get_data(file, train=True):
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # Generate training data from triplets
    train_dataset = datasets.ImageFolder(root="./dataset/", transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('./jbrodbec/task3/embeddings.npy')

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i].split('\\')[-1]] = embeddings[i]
    X = []
    y = []

    # Use the individual embeddings to generate features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack(emb[0], emb[1], emb[2]))
        y.append(1)

        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack(emb[0], emb[2], emb[1]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Function to create data loader from NumPy arrays
def create_loader_from_np(X, y=None, train=True, batch_size=64, shuffle=True, num_workers=4):
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    return loader

# Custom neural network model
class Net(nn.Module):
    def __init__(self):
        super().__init()
        self.fc1 = nn.Linear(6144, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1) 
        self.sigmoid = nn.Sigmoid()
        self.weight_decay = 1e-7

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x)

    def loss(self, prediction, target):
        # Calculate the binary cross-entropy loss
        bce_loss = nn.BCELoss()
        loss = bce_loss(prediction, target)

        # Add L2 regularization to the loss
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param)
        loss += self.weight_decay * l2_loss

        return loss

# Function to train the model
def train_model(train_loader, validation_loader):
    # Initialize the model
    model = Net()
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 10
    prevLoss = 1.0
    prevModel = Net()

    for epoch in range(n_epochs):
        # Train the model using training data
        model.train()
        for [X, y] in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model.forward(X)
            loss = model.loss(y_pred.squeeze(1), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate the loss using validation data
        model.eval()
        totalLoss = 0
        with torch.no_grad():
            for [X, y] in validation_loader:
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                totalLoss += model.loss(y_pred.squeeze(1), y.float()).item() * X.size(0)
        avgLoss = totalLoss / len(validation_loader.dataset)
        print(f'Epoch: {epoch:2}  average validation loss: {avgLoss:10.8f}')
        if prevLoss < avgLoss:
            break
        prevModel = model

    return prevModel

# Function to test the model
def test_model(model, loader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    predictions = []

    # Iterate over the test data
    with torch.no_grad():
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("./jbrodbec/task3/results2.txt", predictions, fmt='%i')

# Main function
if __name__ == '__main__':
    TRAIN_TRIPLETS = './jbrodbec/task3/train_triplets.txt'
    TEST_TRIPLETS = './jbrodbec/task3/test_triplets.txt'

    # Generate embedding for each image in the dataset
    if not os.path.exists('./jbrodbec/task3/embeddings.npy'):
        generate_embeddings()

    # Load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, train_size=0.9)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X_train, y_train, train=True, batch_size=64)
    validation_loader = create_loader_from_np(X_val, y_val, train=True, batch_size=2048)
    test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False)
    
    # Use full data for training if specified
    useFullDataToTrain = True
    if useFullDataToTrain:
        full_loader = create_loader_from_np(X, y, train=True, batch_size=64)
        train_loader = full_loader
    
    # Train the model and test it
    model = train_model(train_loader, validation_loader)
    test_model(model, test_loader)
    print("Results saved to results2.txt")
