import torch.nn as nn
import torch.nn.functional as F

class LightDiscriminator(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(LightDiscriminator, self).__init__()
		self.depth1 = nn.Conv2d(num_classes, num_classes, kernel_size=4, stride = 2, padding=1, groups=num_classes)
		self.point1 = nn.Conv2d(num_classes, ndf, kernel_size=1) 
		self.depth2 = nn.Conv2d(ndf, ndf, kernel_size=4, stride = 2, padding=1, groups=ndf)
		self.point2 = nn.Conv2d(ndf, ndf * 2, kernel_size=1) 
		self.depth3 = nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride = 2, padding=1, groups=ndf * 2)
		self.point3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1) 
		self.depth4 = nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride = 2, padding=1, groups=ndf * 4)
		self.point4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=1)

		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.depth1(x)
		x = self.point1(x)
		x = self.leaky_relu(x)
		x = self.depth2(x)
		x = self.point2(x)
		x = self.leaky_relu(x)
		x = self.depth3(x)
		x = self.point3(x)
		x = self.leaky_relu(x)
		x = self.depth4(x)
		x = self.point4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)

		return x

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)

		return x