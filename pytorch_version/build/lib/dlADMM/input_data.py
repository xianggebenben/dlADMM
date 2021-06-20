# coding: utf-8

import torch
import torchvision
from keras.utils.np_utils import to_categorical
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class mnist():

	def __init__(self):
		self.train_loader = torch.utils.data.DataLoader(
  		torchvision.datasets.MNIST('/data/mnist', train=True, download=True,
                             transform=torchvision.transforms.ToTensor()),
  		batch_size=60000, shuffle=False)
		self.test_loader = torch.utils.data.DataLoader(
			torchvision.datasets.MNIST('/data/mnist', train=False, download=True,
									   transform=torchvision.transforms.ToTensor()),
			batch_size=10000, shuffle=False)
		self.x_train =torch.tensor(self.train_loader.dataset.data,device=device)
		self.y_train =self.train_loader.dataset.targets
		self.x_test =torch.tensor(self.test_loader.dataset.data,device=device)
		self.y_test =self.test_loader.dataset.targets
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

		self.x_train = self.x_train.reshape(60000, 28 * 28)[:55000]
		self.y_train = torch.tensor(to_categorical(self.y_train, num_classes=10).reshape(60000, 10)[:55000],device=device)

		self.x_test = self.x_test.reshape(10000, 28 * 28)
		self.y_test = torch.tensor(to_categorical(self.y_test, num_classes=10).reshape(10000, 10),device=device)

class fashion_mnist():
	def __init__(self):
		self.train_loader = torch.utils.data.DataLoader(
			torchvision.datasets.FashionMNIST('/data/FashionMNIST', train=True, download=True,
									   transform=torchvision.transforms.ToTensor()),
			batch_size=60000, shuffle=False)
		self.test_loader = torch.utils.data.DataLoader(
			torchvision.datasets.FashionMNIST('/data/FashionMNIST', train=False, download=True,
									   transform=torchvision.transforms.ToTensor()),
			batch_size=10000, shuffle=False)
		self.x_train = torch.tensor(self.train_loader.dataset.data,device=device)
		self.y_train = self.train_loader.dataset.targets
		self.x_test = torch.tensor(self.test_loader.dataset.data,device=device)
		self.y_test = self.test_loader.dataset.targets
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

		self.x_train = self.x_train.reshape(60000, 28 * 28)
		self.y_train = torch.tensor(to_categorical(self.y_train, num_classes=10).reshape(60000, 10),device=device)

		self.x_test = self.x_test.reshape(10000, 28 * 28)
		self.y_test = torch.tensor(to_categorical(self.y_test, num_classes=10).reshape(10000, 10),device=device)