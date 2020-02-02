### Criminal-Identification-from-Mugshot-Sketches
> Bachelor Thesis written by Kamaladdin Ahmadzada


### Network architecture scheme
![cap](https://user-images.githubusercontent.com/39130214/73599229-af618b00-455a-11ea-8e94-5ac62e8af97c.png)
CIMSI project combines two different deep neural network architecture itself. First one is [Pix2Pix](https://arxiv.org/pdf/1611.07004) which belongs to Conditional Generative Adversarial Network family.Gets sketch file as a input and this network returns realistic-looking criminal photo. *This should be noted here that all these sketch files must be as much similar as to criminal.Otherwise confident rate drops.In second part of the network given input as a GAN result passes to Siamese net which is widely used for high accuracy one-shot learning technique.Siamese net is a good result of transfer learning in this project ResNet50 architecture used for classification but all this network's training performed by Oxford University. Pre-trained model can be found [here]([https://github.com/ox-vgg/vgg_face2](https://github.com/ox-vgg/vgg_face2)) .In vgg-face2 whole dataset can be found but it is quite large ~40GB that is why pre-trained model preferred .

> It took ~4 hours to get reasonable results like images below.Hardware Specs:
* 1 Tesla T4, GPU
* Architecture:        x86_64
* CPU op-mode(s):      32-bit, 64-bit
* Byte Order:          Little Endian
* CPU(s):              2
* On-line CPU(s) list: 0,1
* Thread(s) per core:  2
* Core(s) per socket:  1
* Socket(s):           1
* NUMA node(s):        1
* Vendor ID:           GenuineIntel
* CPU family:          6
* Model:               63
* Model name:          Intel(R) Xeon(R) CPU @ 2.30GHz
* Stepping:            0
* CPU MHz:             2300.000
* BogoMIPS:            4600.00
* Hypervisor vendor:   KVM
* Virtualization type: full
* L1d cache:           32K
* L1i cache:           32K
* L2 cache:            256K
* L3 cache:            46080K
* NUMA node0 CPU(s):   0,1

> Here are some GAN result 1 :
>  ![2wer](https://user-images.githubusercontent.com/39130214/73605245-5246f300-45b5-11ea-9726-0725b11fde37.png)

>  ![3wer](https://user-images.githubusercontent.com/39130214/73605253-6a1e7700-45b5-11ea-8a26-a8f3228bd585.png)

>  ![4wer](https://user-images.githubusercontent.com/39130214/73605256-77d3fc80-45b5-11ea-939d-e0cf01b732f7.png)










