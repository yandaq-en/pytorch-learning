import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from custom_dataset import CatsAndDogsDataset

my_transforms = transforms.ToTensor()
dataset = CatsAndDogsDataset(csv_file="../dataset/CATDOG/labels.csv", root_dir="../dataset/CATDOG/train", transforms=my_transforms)
