import torch
import torch.nn as nn
import torchvision.models as models


class PreGModel(nn.Module):
    def __init__(self, output_dim):
        super(PreGModel, self).__init__()
        
        # Load pre-trained ResNet-152 model
        resnet = models.resnet152(pretrained=True)
        

        # Remove the final fully connected layer (classification head)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Removing the last fully connected layer
        
        # Add a new fully connected layer with desired output dimensions
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)  # Custom output layer
        
    def forward(self, x):
        # Extract features from ResNet backbone (without the final classification layer)
        x = self.features(x)
        
        # Flatten the output from the ResNet feature extractor
        x = x.view(x.size(0), -1)  # Flatten the output to pass into the new FC layer
        
        # Pass through the custom fully connected layer
        x = self.fc(x)
        return x 
