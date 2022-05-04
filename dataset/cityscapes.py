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

class Map:
    """
    Maps every pixel to the respective object in the dictionary
    Input:
        mapper: dict, dictionary of the mapping
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, input):
        return np.vectorize(self.mapper.__getitem__, otypes=[np.float32])(input)

class Map2:
    """
    Maps every pixel to the respective object in the dictionary
    Input:
        mapper: dict, dictionary of the mapping
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, input):
        return np.array([[self.mapper[element] for element in row]for row in input], dtype=np.float32)

class ToTensor:
    """
    Convert into a tensor of float32: differently from transforms.ToTensor() this function does not normalize the values in [0,1] and does not swap the dimensions
    """
    def __call__(self, input):
        return torch.as_tensor(input, dtype=torch.float32)


class ToNumpy:
    """
    Convert into a tensor into a numpy array
    """
    def __call__(self, input):
        return input.numpy()

# Don't know if it will be useful or if we will subtract the mean inside the dataset class
class MeanSubtraction:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, input):
        return input - self.mean


class Cityscapes(VisionDataset):
    def __init__(self, root, image_folder, train=True, info_file=None, transforms=transforms.ToTensor()):
        """
        Inputs:
            root: string, path of the root folder where images and labels are stored
            list_path: string, path of the file used to split the dataset (train.txt/val.txt)
            image_folder: string, path of the images folder
            transform: transformation to be applied on the images
            target_transform: transformation to be applied on the labels

        self.images = list containing the paths of the images 
        """
        super().__init__(root, transforms)

        self.list_path = "train.txt" if train else "val.txt"                              # path to train.txt/val.txt
        info = json.load(open(f"{root}/{info_file}")) 
        self.train = train          
        self.mapper = dict(info["label2train"])
        self.mean = info["mean"]
        images_folder_path = Path(self.root) / image_folder     # absolute path of the folder containing the images
        

        #Retrive the file names of the images and labels contained in the indicated folders
        image_name_list = np.array(sorted(images_folder_path.glob("*")))

        #Prepare lists of data and labels
        name_samples = [l.split("/")[1] for l in np.loadtxt(f"{root}/{self.list_path}", dtype="unicode")] # creates the list of the images names for the train/validation according to list_path
        self.images = [img for img in image_name_list if str(img).split("/")[-1] in name_samples]    # creates the list of images names filtered according to name_samples
        


    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        image = np.array(Image.open(image_path), dtype=np.float32)

        
        image = MeanSubtraction(self.mean)(image)
        
        if self.transforms and self.train:
            seed = np.random.randint(10000)
            torch.manual_seed(seed)
            image = self.transforms(image)    # applies the transforms for the images
        else:
            image = transforms.ToTensor()(image)
        
        return image

def printImageLabel(image, label):
    info = json.load(open("/Users/gio/Documents/GitHub/BiSeNet/data/Cityscapes/info.json"))
    mean = torch.as_tensor(info["mean"])
    image = (image.permute(1, 2, 0) + mean).permute(2, 0, 1)
    mapper = {i if i!=19 else 255:info["palette"][i] for i in range(20)}
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    composed = torchvision.transforms.Compose([ToNumpy(), Map2(mapper), transforms.ToTensor(), transforms.ToPILImage()])
    axs[0].imshow(transforms.ToPILImage()(image.to(torch.uint8)))
    axs[1].imshow(composed(label))
    plt.show()

if __name__ == "__main__":
    crop_width = 1024
    crop_height = 512
    composed = torchvision.transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(0, scale=[0.75, 2.0]), transforms.RandomCrop((crop_height, crop_width), pad_if_needed=True), transforms.GaussianBlur(kernel_size=3)])
    data = Cityscapes("./data/Cityscapes", "images/", train=True, info_file="info.json", transforms=composed)
    image = data[5]
    printImageLabel(image)