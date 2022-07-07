'''
	added to enable loading models using torch.hub
'''

import os
import torch
import torchvision

import models.gc_vit as _gc_vit

dependencies = ['torch', 'torchvision']

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform
  
def gc_vit_tiny_in1k(pretrained=True, **kwargs):
	"""
	Global Context Vision Transformer (GC ViT Tiny)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _gc_vit.gc_vit_tiny()
	if pretrained:
		checkpoint_url = "https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/gcvit/gcvit_tiny_best-15a2241d.pth.tar"
		cache_file_name = "gcvit_tiny_best-15a2241d.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '15a2241d'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform  
