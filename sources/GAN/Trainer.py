import GAN.networks as nets
import Utils.misc as misc
import yaml
import torch
from  torch.autograd import Variable

class Trainer():
    def __init__(self):
        parameters           =yaml.safe_load(open("parameters.yaml","r"))["parameters"]
        self.epoch           =int(parameters["epoch"])
        lr              =int(parameters["lr"])
        data_path            =parameters["data_path"]
        self.hardware        =parameters["hardware"]
        betas                =(int(parameters["beta1"]),int(parameters["beta2"]))
        self.lambda_         =int(parameters["lambda"])
        test_path            =parameters["test_path"]

        self.gen             =nets.GENERATOR()
        self.dis             =nets.DISCRIMINATOR()
        self.dataloader      =misc.dataloader_initializer(data_path,bsize=int(parameters["batch_size"]))
        self.test_dataloder  =misc.dataloader_initializer(test_path)
        #self.gen.weight_initializer(0.5,0.5)
        #self.dis.weight_initializer(0.5,0.5)

        self.criterion_bce=torch.nn.BCELoss()
        self.criterion_l1=torch.nn.L1Loss()
        self.dis_optimizer=torch.optim.Adam(self.dis.parameters(),lr=lr,betas=betas)
        self.gen_optimizer=torch.optim.Adam(self.gen.parameters(),lr=lr,betas=betas)
        
    def start(self):
        counter=0
        if(self.hardware=="gpu"):
            self.gen=self.gen.cuda()
            self.dis=self.dis.cuda()
        
        for count in range(self.epoch):
            for image, _ in self.dataloader:
                self.gen.zero_grad() 
                real_out = image[:, :, :, 0:256]
                input = image[:, :, :, 256:]
        
                if(self.hardware=="gpu"):
                    input, real_out = Variable(input.cuda()), Variable(real_out.cuda())
                
                dis_out = self.dis(input, real_out).squeeze()

                if(self.hardware=="gpu"):
                    dis_loss_real = self.criterion_bce(dis_out, Variable(torch.ones(dis_out.size()).cuda()))
                else:
                    dis_loss_real = self.criterion_bce(dis_out, Variable(torch.ones(dis_out.size())))
        
                gen_out = self.gen(input)
                dis_out = self.dis(input, gen_out).squeeze()

                if(self.hardware=="gpu"):
                    dis_loss_fake = self.criterion_bce(dis_out, Variable(torch.zeros(dis_out.size()).cuda()))
                else:
                    dis_loss_fake = self.criterion_bce(dis_out, Variable(torch.zeros(dis_out.size())))        

                dis_general_loss = (dis_loss_fake + dis_loss_real) * 0.5
                dis_general_loss.backward()
                self.dis_optimizer.step()

                self.gen.zero_grad()
                gen_out = self.gen(input)
                dis_out = self.dis(input, gen_out).squeeze()

                if(self.hardware=="gpu"):
                    gen_loss = self.criterion_bce(dis_out, Variable(torch.ones(dis_out.size()).cuda())) + self.lambda_ * self.criterion_l1(gen_out, real_out)
                else:
                    gen_loss = self.criterion_bce(dis_out, Variable(torch.ones(dis_out.size()))) + self.lambda_ * self.criterion_l1(gen_out, real_out)

                gen_loss.backward()
                self.gen_optimizer.step()

                counter+=1
                print("Epoch {} completed ... ".format(epoch))
                if(counter%20==0):
                    img=next(iter(self.test_dataloder))
                    print("Iteration Count : {}".format(counter))
                misc.show_tensor(gen(Variable(img[0]).to(device="gpu" if (self.hardware=="gpu") else "cpu", dtype=torch.float)))

        print("Saving all model parameters...")
        torch.save(self.gen.state_dict(), './gen.pkl')
        torch.save(self.dis.state_dict(), './dis.pkl')

