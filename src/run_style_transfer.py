#main code for running style transfer between style and content images 
import process_images
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from loss_functions import GramMatrix_gen, GramMSELoss
from torch import optim
from network import model_instantiate
import numpy as np
import argparse
import os

def initialize_image_tensors(style_image_path,content_image_path,device):
	"""Generate the image tensors for style and content images and the variable images which will be worked upon
	"""

	prep=process_images.preprocess_transforms()

	style_image_obj=Image.open(style_image_path)
	content_image_obj=Image.open(content_image_path)

	style_image_prep=prep(style_image_obj)
	content_image_prep=prep(content_image_obj)

	if (device =='cuda'):
		style_image_torch=Variable(style_image_prep.unsqueeze(0).cuda())
		content_image_torch=Variable(content_image_prep.unsqueeze(0).cuda())
		#copying the content preprocessed image to the variable image which we will be worked upon
		var_image_torch=Variable(content_image_torch.data.clone(),requires_grad=True)
	else:
		style_image_torch=Variable(style_image_prep.unsqueeze(0))
		content_image_torch=Variable(content_image_prep.unsqueeze(0))
		var_image_torch=Variable(content_image_torch.clone(),requires_grad=True)

	return(style_image_torch,content_image_torch,var_image_torch)



def style_transfer_main(network,style_image,content_image,var_image,network_option="vgg16",max_epochs=100,device="cpu"):
	"""main function for running the optimization of the style transfer algorithm
	"""

	#weight initialization and style content layer initialization for the network
	if(network_option == "vgg16"):
			style_layers=["relu1_2", "relu2_2", "relu3_3","relu4_3","relu5_1"]
			content_layers=["relu4_2"]
			style_weight_list=[1e3/n**2 for n in [64,128,256,512,512]] #5 weights for 5 layers
			content_weight_list=[1e0]

			style_weights=dict(zip(style_layers,style_weight_list))
			content_weights=dict(zip(content_layers,content_weight_list))

	style_feature=network(style_image,option="style") #a dictionary with each key as the style layer name
	content_feature=network(content_image,option="content") # a dictionary with each key as the content layer name 


	style_target={}
	content_target={}

	#generating the style_target items
	for key,value in style_feature.items():
		style_target[key]=GramMatrix_gen()(style_feature[key]).detach()

	#generating the content_target items
	for key,value in content_feature.items():
		content_target[key]=content_feature[key].detach()

	#initializing style and content loss functions 
	style_loss_fns=[GramMSELoss()]*len(style_layers)
	content_loss_fns=[nn.MSELoss()]*len(content_layers)

	if device=="cuda":
		style_loss_fns = [style_loss_fn.cuda() for style_loss_fn in style_loss_fns]
		content_loss_fns = [content_loss_fn.cuda() for content_loss_fn in content_loss_fns]
    

	style_loss_fns=dict(zip(style_layers,style_loss_fns))
	content_loss_fns=dict(zip(content_layers,content_loss_fns))

	optimizer=optim.LBFGS([var_image])

	print(max_epochs)
	num_iter=[0]
	while num_iter[0] <= max_epochs:
		print(num_iter[0])
		def closure():
			optimizer.zero_grad()
    		#compute for the variable image style and content loss components
			style_var_feature=network(var_image,option="style")
			content_var_feature=network(var_image,option="content")

    		#style_loss_function compute
			style_losses=[ style_weights[key]*style_loss_fns[key](style_var_feature[key],value) for key,value in style_target.items() ]

    		#content loss_function compute 
			content_losses=[ content_weights[key]*content_loss_fns[key](content_var_feature[key],value) for key,value in content_target.items() ]

    		#overall loss 
			style_loss=sum(style_losses)
			content_loss=sum(content_losses)

			loss=style_loss+content_loss 
			loss.backward()
			num_iter[0]=num_iter[0]+1
			print('Iteration: %d, Style loss: %f, Content loss: %f, Overall loss:%f '%(num_iter[0]+1, style_loss, content_loss, loss))
			
			return(loss)
		optimizer.step(closure)
	return(var_image)


if __name__ == '__main__':
	
	
	#parsing the command line arguments for getting the filepaths
	parser=argparse.ArgumentParser()
	parser.add_argument('--style_path', type=str, 
        help='Path of the style image', default='../images/wheat_field_van_gogh.jpg')
	parser.add_argument('--content_path', type=str, 
        help='Path of the content image', default='../images/santa_monica.jpg')
	parser.add_argument('--result_dir', type=str, 
        help='Folder for the output results', default='../images')
	parser.add_argument('--max_epochs', type=int, 
        help='Maximum number of epochs', default=50)
	parser.add_argument('--device', type=str, 
        help='Device option', default="cpu")
	args=parser.parse_args()
	

	style_image_torch,content_image_torch,var_image_torch=initialize_image_tensors(args.style_path,args.content_path,args.device)
	loss_network=model_instantiate(device=args.device)
	print(loss_network)
	var_image=style_transfer_main(loss_network,style_image_torch,content_image_torch,var_image_torch,max_epochs=args.max_epochs,device=args.device)

	if (args.device=="cpu"):
		post=process_images.postprocess_tensor(var_image.data[0].squeeze())
	else:
		post=process_images.postprocess_tensor(var_image.data[0].cpu().squeeze())
	post.save(os.path.join(args.result_dir,'results.png'))
	
















