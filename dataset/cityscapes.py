import sys
sys.path.insert(1, "./")

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
    def __init__(self, root, image_folder, labels_folder, train=True, info_file=None, transforms=transforms.ToTensor()):
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
        info = json.load(open(f"{root}/{info_file}")) 
        self.train = train          
        self.mapper = dict(info["label2train"])
        self.mean = info["mean"]
        images_folder_path = Path(self.root) / image_folder     # absolute path of the folder containing the images
        labels_folder_path = Path(self.root) / labels_folder    # absolute path of the folder containing the labels
        

        #Retrive the file names of the images and labels contained in the indicated folders
        image_name_list = np.array(sorted(images_folder_path.glob("*")))
        labels_list = np.array(sorted(labels_folder_path.glob("*")))

        #Prepare lists of data and labels
        name_samples = [l.split("/")[1] for l in np.loadtxt(f"{root}/{self.list_path}", dtype="unicode")] # creates the list of the images names for the train/validation according to list_path
        self.images = [img for img in image_name_list if str(img).split("/")[-1] in name_samples]    # creates the list of images names filtered according to name_samples
        self.labels = [img for img in labels_list if str(img).split("/")[-1].replace("_gtFine_labelIds.png", "_leftImg8bit.png") in name_samples]  # creates the list of label image names filtered according to name_samples
        


    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]

        image = np.array(Image.open(image_path), dtype=np.float32)
        label = np.array(Image.open(label_path), dtype=np.float32)

        
        image = MeanSubtraction(self.mean)(image)
        label = Map(self.mapper)(label)
        print(np.unique(label))

        
        if self.transforms and self.train:
            seed = np.random.randint(10000)
            torch.manual_seed(seed)
            image = self.transforms(image)    # applies the transforms for the images
            torch.manual_seed(seed)
            label = self.transforms(label)    # applies the transforms for the labels
        else:
            image = transforms.ToTensor()(image)
            label = transforms.ToTensor()(label)
        
        return image, label[0]




if __name__ == "__main__":
    crop_width = 1024
    crop_height = 512
    composed = torchvision.transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(0, scale=[0.75, 2.0]), transforms.RandomCrop((crop_height, crop_width), pad_if_needed=True)])
    data = Cityscapes("./data/Cityscapes", "images/", labels_folder='labels/',train=True, info_file="info.json", transforms=composed)
    image, label = data[5]
    
    
    
    #info
    info = json.load(open("./data/Cityscapes/info.json"))

    #Image
    image = transforms.ToPILImage()(image.to(torch.uint8))
    
    #Label
    info = json.load(open("./data/Cityscapes/info.json"))
    palette = {i if i!=19 else 255:info["palette"][i] for i in range(20)}
    label = colorLabel(label,palette)


    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[1].imshow(label)
    axs[1].axis('off')
    plt.show()


    
    #transforms.ToPILImage()(image_label.to(torch.uint8)).show()



