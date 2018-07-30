#code for instantiating the network in pytorch 
from torchvision import models, transforms
import torch
from PIL import Image
from torch.autograd import Variable
#current default option being VGG 
#later can be used for other networks also 



class VGGLoSSNetwork(torch.nn.Module):
	def __init__(self,vgg_model):
		super(VGGLoSSNetwork,self).__init__()
		self.layer_set=vgg_model.features
		self.style_name_maps={
		 '3': "relu1_2",
		 '8': "relu2_2",
		 '15': "relu3_3",
		 '22': "relu4_3",
		 '24': "relu5_1"
		}
		self.content_name_maps={
		'20': "relu4_2"
		}
	def forward(self,x,option="style"):
		output={}
		if option == "style":
			for name,module in self.layer_set._modules.items():
				x=module(x)
				if name in self.style_name_maps:
					output[self.style_name_maps[name]]=x
		elif option == "content":
			for name,module in self.layer_set._modules.items():
				x=module(x)
				if name in self.content_name_maps:
					output[self.content_name_maps[name]]=x
		return(output)

def model_instantiate(model_option='vgg16'):

	"""Instantiates the pretrained models along with the networks for various loss computations
	"""

	if(model_option=='vgg16'):
		model_vgg=models.vgg16(pretrained=True) 
		for param in model_vgg.parameters():
			param.requires_grad=False
		Lossnetwork=VGGLoSSNetwork(model_vgg)
		return(Lossnetwork)

def forward_pass_example(img_path):
	"""Run a forward pass through the network of type Lossnetwork and generate the style and content components for an image
	   a test function in general
	"""
	normalize = transforms.Normalize(
   		mean=[0.485, 0.456, 0.406],
   		std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
   		transforms.Scale(256),
   		transforms.CenterCrop(224),
   		transforms.ToTensor(),
   		normalize
	])
	img_pil=Image.open(img_path)
	img_tensor = preprocess(img_pil)
	img_tensor.unsqueeze_(0)
	img_variable = Variable(img_tensor)

	return(img_variable)


if __name__ == '__main__':
	img_path="C:\\Users\\bosed\\Documents\\style_transfer_codes\\content_images\\emma_watson_image_1.jpg"
	img_content=forward_pass_example(img_path)
	loss_network=model_instantiate()
	op=loss_network(img_content,option="style")
	print(op)












