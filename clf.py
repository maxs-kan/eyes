import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import joblib
import argparse
import imageio

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(576, 256)
        self.fc_mu = nn.Linear(256, 25)
        self.fc_logvar = nn.Linear(256, 25)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class OpenEyesClassificator():
    def __init__(self):
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load('./encoder.pth', map_location='cpu'))
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.clf = joblib.load('./classifier.pkl')
        self.scaler = joblib.load('./scaler.pkl')
    
    def prepare_data(self, path):
        img = imageio.imread(path).astype(np.float32).reshape(-1) / 255
        return img
    
    def predict(self, inpIm):
        img = self.prepare_data(inpIm)
        img, _ = self.encoder(torch.from_numpy(img))
        img = self.scaler.transform(img.numpy().reshape(1, -1))
        score = self.clf.predict_proba(img)
        return score[0, 1]
