#function for preprocessing images 

#Preprocessing Steps involved: 
"""
1.Resize to 512 x 512
2. Convert to Tensor
3. Flip channels from RGB to BGR
4. Subtract imagenet mean
5. multilpy each pixel by 255
"""
from torchvision import transforms 
import torch


def preprocess_transforms(img_size=512):
	"""Preprocessing transforms before feeding to network
	"""
	prep=transforms.Compose([transforms.Scale(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
	return(prep)

def postprocess_transforms():
	"""Postprocessing transformations after getting the output from network
	"""
	post = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
	return(post)

def postprocess_tensor(tensor):
	"""postprocess a tensor output
	"""
	post=postprocess_transforms()
	img_rev=post(tensor)
	img_rev[img_rev>1]=1
	img_rev[img_rev<0]=0
	PIL_transform=transforms.Compose([transforms.ToPILImage()])
	img_post_processed=PIL_transform(img_rev)

	return(img_post_processed)




