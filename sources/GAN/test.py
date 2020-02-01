import GAN.networks as nets
import Utils.misc as misc
import yaml
import torch
import torch.autograd.Variable

class Tester():
	def __init__(self):
		parameters      =yaml.safe_load("parameters.yaml")["parameters"]
		genweights      =parameters["pretrained_weights_gen"]
		hardware        =parameters["hardware"]
		test_path       =parameters["test_path"]
		self.out_path	=parameters["output_path"]
		self.gen        =nets.GENERATOR()
		self.gen.load_state_dict(torch.load(genweights))
		self.test_dataloder  =misc.dataloader_initializer(test_path)  
	def start(self):
		for input,_ in test_dataloder:
			output=gen(input)
			for x in range(output.shape[0]):
				misc.show_tensor(output[x])
				misc.save_out(self.out_path+str(x)+".png")
		print("Operation completed ...")

