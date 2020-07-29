import os
import glob
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

# Use this for non-folder wise images
'''
class ImageFolder(data.Dataset):

	def __init__(self, folder_path):
		self.folder_path = folder_path
		self.images_name = sorted(os.listdir(self.folder_path))
		self.images_name = self.images_name[:-1]
		self.targets = [int(name[0:4]) for name in self.images_name]
		self.total_samples = len(self.targets)
		self.totensor = transforms.ToTensor()
		self.horizontal_flip = transforms.RandomHorizontalFlip(p = 0.5)

	def __getitem__(self, index):
		
		img = Image.open(self.folder_path + self.images_name[index]).convert('RGB')
		img = self.horizontal_flip(img)
		img = self.totensor(img)
		target = self.targets[index]
		
		return img, target

	def __len__(self):
		
		return self.total_samples
'''

# Use this for folder wise images

class ImageFolder(data.Dataset):

	def __init__(self, folder_path):
		self.folder_path = folder_path
		self.targets_name = sorted(os.listdir(self.folder_path))
		subfolder_name = []
		images_name = []
		targets = []
		
		for folder_name in self.targets_name:
			path = os.listdir(self.folder_path + folder_name + '/')
			n_images = len(path)
			images_name.extend(path)
			subfolder_name.extend([(folder_name + '/')]*n_images)
			targets.extend([int(folder_name)]*n_images)
		
		self.subfolder_name = subfolder_name
		self.images_name = images_name
		self.targets = targets
		self.total_samples = len(self.targets)
		self.totensor = transforms.ToTensor()
		self.horizontal_flip = transforms.RandomHorizontalFlip(p = 0.5)

	def __getitem__(self, index):
		
		img = Image.open(self.folder_path + self.subfolder_name[index] + self.images_name[index]).convert('RGB')
		img = self.horizontal_flip(img)
		img = self.totensor(img)
		target = self.targets[index]
		
		return img, target

	def __len__(self):
		
		return self.total_samples

class myBatchSampler(data.BatchSampler):

	def __init__(self, sampler, train_dataset, n_classes, n_samples):
		self.sampler = sampler
		self.train_dataset = train_dataset
		self.n_classes = n_classes
		self.n_samples = n_samples
		self.batch_size = self.n_classes * self.n_samples
		self.total_samples = len(self.sampler)
		self.targets = self.train_dataset.targets
		self.targets = np.array(self.targets)
		self.labels_set = set(self.targets)
		self.labels_to_indices = {label: np.where(self.targets == label)[0] for label in self.labels_set}
		self.classwise_used_label_to_indices = {label: 0 for label in self.labels_set}

	def __iter__(self):
		
		self.count = 0
		while(self.count + self.batch_size < self.total_samples):
			self.count = self.count + self.batch_size
			labels_chosen = np.random.choice(list(self.labels_set), self.n_classes, replace = False)
			batch_images_indices = []
			for label in labels_chosen:
				indices = self.labels_to_indices[label][self.classwise_used_label_to_indices[label]:
				(self.classwise_used_label_to_indices[label] + self.n_samples)]
				if len(indices) < 4:
					indices_to_add = 4 - len(indices)
					new_indices = list(indices)
					for i in range(indices_to_add):
						new_indices.append(indices[i])
					indices = new_indices
				batch_images_indices.extend(indices)
				self.classwise_used_label_to_indices[label] += self.n_samples
				if self.classwise_used_label_to_indices[label] + self.n_samples > len(self.labels_to_indices[label]):
					np.random.shuffle(self.labels_to_indices[label])
					self.classwise_used_label_to_indices[label] = 0

			yield batch_images_indices

	def __len__(self):

		return self.total_samples // self.batch_size
