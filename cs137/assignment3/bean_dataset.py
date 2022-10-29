import os
import random
from torch.utils.data import Dataset
from torchvision.io import read_image


def list_images(folder): 
    """
    This function list all `.jpg` images in a folder and return paths of these images in a list.
    args:
        folder: string, a path 
    returns: 
        a list of paths. For example, if an image `img0.jpg` is in the folder, then the path of 
        the image is `folder + "/img0.jpg"`

    """

    res = []
    for file in os.listdir(folder):
        # check only text files
        if file.endswith('.jpg'):
            res.append(os.path.join(folder, file))

    return res



class BeanImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):

        # TODO: read in all image paths from `img_dir`. In this assignment, `img_dir` should be `data/train`, `data/validation`, or `data/test`

        # TODO: according to the subfolder ("angular_leaf_spot", "bean_rust", or "healthy"), assign labels (0, 1, or 2) to these images 
        
        # TODO: store all images and their labels to member variables 
        

    def __len__(self):

        # TODO: get the size of the dataset and return it. 
        return None

    def __getitem__(self, idx):

        # TODO: locate the path and label of the `idx`-th image; read in the image to the **float** tensor `image`; and assign its label to `label`

        image = None
        label = None

        # apply necessary transformations if necessary
        # TODO: read "https://pytorch.org/vision/stable/transforms.html" to find more details

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
