import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from networks import *
from loss_functions import *
from datasets import *

writer = SummaryWriter('logs_market1501_batchall_softplus/person_reid')
images_path = '../../market_1501/'
os.makedirs('checkpoints_market1501_batchall_softplus', exist_ok = True)
n_classes = 18
n_samples = 4
batch_size = n_classes * n_samples
train_dataset = ImageFolder(folder_path = images_path)
mySampler = SequentialSampler(train_dataset)
myBatchSampler = myBatchSampler(mySampler, train_dataset, n_classes = n_classes, n_samples = n_samples)
train_loader = DataLoader(train_dataset, shuffle = False, num_workers = 4, batch_sampler = myBatchSampler)

no_of_training_batches = len(train_loader)/batch_size
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 50

embeddingNet = EmbeddingNet()
optimizer = optim.Adam(embeddingNet.parameters(), lr = 3e-5, betas = (0.9, 0.999))

def run_epoch(data_loader, model, optimizer, split = 'train', epoch_count = 0):

	model.to(device)

	if split == 'train':
		model.train()
	else:
		model.eval()

	running_loss = 0.0

	for batch_id, (imgs, labels) in enumerate(train_loader):

		iter_count = epoch_count * len(data_loader) + batch_id
		imgs = imgs.to(device)
		embeddings = model.get_embeddings(imgs)
		embeddings_2_norm = torch.mean(torch.norm(embeddings, p = 2, dim = 1))
		batch_loss, non_zero_losses = batch_all_online_triplet_loss(labels, embeddings, margin = 0.2, squared = False)
		optimizer.zero_grad()
		
		if split == 'train':
			batch_loss.backward()
			optimizer.step()

		running_loss = running_loss + batch_loss.item()

		# Adding the logs in Tensorboard
		writer.add_scalar('Batch_All_Online_Triplet_Loss', batch_loss.item(), iter_count)
		writer.add_scalar('2-norm of Embeddings', embeddings_2_norm ,iter_count)
		writer.add_scalar('% non-zero losses in batch', (non_zero_losses*100), iter_count)

	return running_loss

def fit(train_loader, model, optimizer, n_epochs):

	print('Training Started\n')
	
	for epoch in range(n_epochs):
		
		loss = run_epoch(train_loader, model, optimizer, split = 'train', epoch_count = epoch)
		loss = loss/no_of_training_batches

		print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)
		torch.save({'state_dict': model.cpu().state_dict()}, 'checkpoints_market1501_batchall_softplus/model_epoch_' + str(epoch + 1) + '.pth')

fit(train_loader, embeddingNet, optimizer = optimizer, n_epochs = epochs)