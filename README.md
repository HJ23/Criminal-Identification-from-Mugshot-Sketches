### Criminal-Identification-from-Mugshot-Sketches
> Bachelor Thesis written by Kamaladdin Ahmadzada


### Network architecture scheme
![cap](https://user-images.githubusercontent.com/39130214/73599229-af618b00-455a-11ea-8e94-5ac62e8af97c.png)
CIMSI project combines two different deep neural network architecture itself. First one is [Pix2Pix](https://arxiv.org/pdf/1611.07004) which belongs to Conditional Generative Adversarial Network family.Gets sketch file as a input and this network returns realistic-looking criminal photo. *This should be noted here that all these sketch files must be as much similar as to criminal.Otherwise confident rate drops.In second part of the network given input as a GAN result passes to Siamese net which is widely used for high accuracy one-shot learning technique.Siamese net is a good result of transfer learning in this project ResNet50 architecture used for classification but all this network's training performed by Oxford University. Pre-trained model can be found [here]([https://github.com/ox-vgg/vgg_face2](https://github.com/ox-vgg/vgg_face2)) .In vgg-face2 whole dataset can be found but it is quite large ~40GB that is why pre-trained model preferred .

> Here are some results :
>   









