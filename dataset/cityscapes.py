import sys
sys.path.insert(1, "./")
import os

from cProfile import label
from turtle import color
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision
import json

from utils import Map, Map2, MeanSubtraction, ToNumpy, colorLabel




class Cityscapes(VisionDataset):
    def __init__(self, root, images_folder, labels_folder, train=True, info_file=None, transforms=[]):
        """
        Inputs:
            root: string, path of the root folder where images and labels are stored
            list_path: string, path of the file used to split the dataset (train.txt/val.txt)
            image_folder: string, path of the images folder
            labels_folder: string, path of the labels folder
            transform: transformation to be applied on the images
            target_transform: transformation to be applied on the labels

        self.images = list containing the paths of the images 
        self.labels = list contating the paths of the labels
        """
        super().__init__(root, transforms)
        
        self.list_path = "train.txt" if train else "val.txt"                              # path to train.txt/val.txt
        info = json.load(open(os.path.join(self.root, info_file)))
        self.train = train          
        self.mapper = dict(info["label2train"])
        self.mean = info["mean"]
        self.images_folder_path = Path(os.path.join(self.root, images_folder))     # absolute path of the folder containing the images
        self.labels_folder_path = Path(os.path.join(self.root, labels_folder))    # absolute path of the folder containing the labels
        
        #Retrive the file names of the images and labels contained in the indicated folders
        image_name_list = np.array(sorted(os.listdir(self.images_folder_path)))
        labels_list = np.array(sorted(os.listdir(self.labels_folder_path)))

        #Prepare lists of data and labels
        name_samples = [l.split("/")[1] for l in np.loadtxt(os.path.join(self.root, self.list_path), dtype="unicode")] # creates the list of the images names for the train/validation according to list_path
        self.images = [img for img in image_name_list if str(img) in name_samples]    # creates the list of images names filtered according to name_samples
        self.labels = [img for img in labels_list if str(img).replace("_gtFine_labelIds.png", "_leftImg8bit.png") in name_samples]  # creates the list of label image names filtered according to name_samples


    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return len(self.images)


    def __getitem__(self, index):
        image_path = os.path.join(self.images_folder_path,self.images[index])
        label_path = os.path.join(self.labels_folder_path,self.labels[index])

        image = np.array(Image.open(image_path), dtype=np.float32)
        label = np.array(Image.open(label_path), dtype=np.float32)

        image = MeanSubtraction(self.mean)(image)
        label = Map(self.mapper)(label)

        if self.transforms and self.train:
            seed = np.random.randint(10000)
            torch.manual_seed(seed)
            image = self.transforms(image)    # applies the transforms for the images
            torch.manual_seed(seed)
            label = self.transforms(label)    # applies the transforms for the labels
        else: #FA IL RESIZE SE NON SIAMO IN TRAIN? NON PER FORZA RANDOM CROP
            image = transforms.ToTensor()(image)
            image = transforms.Resize((512, 1024))(image) 
            label = transforms.ToTensor()(label)
            label = transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST)(label)

        return image, label[0]




if __name__ == "__main__":
    crop_width = 1024
    crop_height = 512
    composed = torchvision.transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop((crop_height, crop_width), pad_if_needed=True)])
    data = Cityscapes(root=os.path.join(os.getcwd(),"data","Cityscapes"), images_folder="images", labels_folder="labels", train=True, info_file="info.json") #transforms=composed
    image, label = data[5]
    #print(image.size()) to see if resize work
    print(os.getcwd())
    #info
    info = json.load(open(os.path.join(os.getcwd(),"data","Cityscapes","info.json")))

    #Image
    image = transforms.ToPILImage()(image.to(torch.uint8))
    
    #Label
    palette = {i if i!=19 else 255:info["palette"][i] for i in range(20)}
    label = colorLabel(label,palette)

    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[1].imshow(label)
    axs[1].axis('off')
    plt.show()


    
    #transforms.ToPILImage()(image_label.to(torch.uint8)).show()



