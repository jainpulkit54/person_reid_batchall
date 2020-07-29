import torch.nn as nn
from torchvision import models

class EmbeddingNet(nn.Module):

	def __init__(self):
		super(EmbeddingNet, self).__init__()
		resnet50 = models.resnet50(pretrained = True, progress = True)
		resnet50 = list(resnet50.children())[:-1]
		self.conv_layers = nn.Sequential(*resnet50)
		self.fc_layers = nn.Sequential(
			nn.Linear(2048, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 128)
			)

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(x.shape[0],-1)
		x = self.fc_layers(x)
		return x

	def get_embeddings(self, x):
		return self.forward(x)