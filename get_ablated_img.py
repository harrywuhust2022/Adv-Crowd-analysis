import torch
import numpy as np
import cv2
from utils_mean import *
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def array_transpose(array):
    temp = np.transpose(array,(1,2,0))
    return temp


#img = cv2.imread('./IMG_26.jpg', 1)
#img_tensor = torch.from_numpy(img)


img = Image.open('./IMG_26.jpg')

transform = transforms.ToTensor()
img_tensor = transform(img)

print("img_tensor_shape: ", img_tensor.shape)
print("img_tensor_type: ", type(img_tensor))

clean_ablated = random_mask_batch_one_sample(img_tensor.cuda(), 5000, reuse_noise=False)
ablated_img = clean_ablated.cpu().numpy()

ablated_img = array_transpose(ablated_img)
ablated_img = normalize(ablated_img)

plt.imsave('./ablated_IMG_26.png', ablated_img, format='png', cmap=plt.cm.jet)
