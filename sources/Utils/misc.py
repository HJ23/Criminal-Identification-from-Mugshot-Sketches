from matplotlib import pyplot
import torchvision.transforms as transform
import torchvision.datasets as dtst
import torch.utils as utils
import torch.nn as nn
# Parameter_Parser can be implemented but in this project it  omitted functionality


def dataloader_initializer(path,bsize=10):
    transformation=transform.Compose([transform.ToTensor(),transform.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    traindata=dtst.ImageFolder(root=path,transform=transformation)
    traindataloader=utils.data.DataLoader(traindata,batch_size=bsize,shuffle=True,num_workers=4)
    return traindataloader

def winit(layer, mean, std):
  # normalize all inputs to uniform distribution
    if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(mean, std)
        layer.bias.data.zero_()

def save_out(path,tensor):
    tensor=(tensor.cpu().numpy().transpose(1, 2, 0) + 1) / 2
    pyplot.savefig(path,bbox_inches='tight', pad_inches=0,transparent=True)
    
# utility for displaying tensor as image 
def show_tensor(tensor):
    pyplot.imshow((tensor.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    pyplot.show()
