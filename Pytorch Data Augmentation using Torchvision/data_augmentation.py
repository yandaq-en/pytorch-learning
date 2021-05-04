import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from custom_dataset import CatsAndDogsDataset

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.8),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.5),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.,0.,0.],std=[1.,1.,1.])
])

dataset = CatsAndDogsDataset(csv_file="../dataset/CATDOG/sample.csv", root_dir="../dataset/CATDOG/sample", transform=my_transforms)

img_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, "img"+str(img_num)+".png")
        img_num += 1