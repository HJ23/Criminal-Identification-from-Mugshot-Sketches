import sys
sys.path.append("./sources")

import yaml
from GAN.Trainer import Trainer
from GAN.Tester import Tester
from SiamaseNET.SiamNET import start


parameters	=   yaml.safe_load(open("parameters.yaml","r"))["parameters"]
tobj=None

try:    
    train_or_test	=True if(parameters["train_or_test_GAN"]=="train") else False
    
    if(train_or_test):
        tobj=Trainer()
        tobj.start()
    
    tobj=Tester()
    tobj.start()
    start(parameters["police_database"],"./out/GAN_results/1.png")

except Exception as exp:
    print(exp)