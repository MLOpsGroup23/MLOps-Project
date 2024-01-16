# -------------------------------------------------------------------------------------- #
# ----- This file is used to take random samples from the Fashion MNIST Test set ------- #
# ----- And convert it to a BMP image.  ------------------------------------------------ #
# ----- This can later be uploaded to our server for the model to process -------------- #
# -------------------------------------------------------------------------------------- #

import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

def tensor_to_bmp(tensor_img, img_name): # Input: (3, 28, 28) tensor image or (1,28,28)
    # Convert from a 3 channel image to a single channel image
    if(tensor_img.shape[0] == 3):
        tensor_img = tensor_img[0].unsqueeze(0) # Convert from (3,28,28) to (1,28,28)
    # The tensor needs to be converted from [min, max] into [0, 255], as the .bmp image must be 8-bit
    tensor_img = tensor_img - tensor_img.min() 
    tensor_img = tensor_img / tensor_img.max() 
    tensor_img = tensor_img * 255              
    # Convert to 8-bit integer
    tensor_img = tensor_img.type(torch.uint8)

    image = tensor_img.squeeze(0) # Reduce from 

    image = Image.fromarray(image.numpy(), mode='L')
    image.save('./reports/images/' + img_name + ".bmp")



def create_random_fashion_img(datapath, img_name):
    data = torch.load(datapath)
    dataset  = TensorDataset(data[0], data[1])  # create your datset
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    dataiter = iter(dataloader)
    test_images, _ = next(dataiter)
    tensor_to_bmp(test_images[0], img_name)

if __name__ == '__main__':
    create_random_fashion_img("./data/processed/test.pt", "extractedImg")
    