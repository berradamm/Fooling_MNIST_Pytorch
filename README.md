# MNIST_Pytorch
## ML's Hello World
First version is a classical MNIST with fully connected Layer
## Result
![mnist](https://user-images.githubusercontent.com/45148200/49903347-35a6a380-fe67-11e8-8a21-8873d2150336.PNG)

## Adverserial Images 
Here is a basic notebook summerizing how to fool a classical MNIST
### Train Model 
So first we either use a pretrained MNIST or we retrain an model from scratch which is not so complicated
### Fast Gradient Step Method - Attack
We are going to apply the following function that aims to perturbate the original inputs to our inputs:

![codecogseqn 1](https://user-images.githubusercontent.com/45148200/49902351-ff1b5980-fe63-11e8-873e-5fb16f8a99ce.gif)

And here is the function in python 
``` python 
# FGSM
def fgsm (image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
```
### Test Model
This the main part of this application because its in the test function that we are gonna apply our fooling function.
In other words, each test sample,  the function computes the gradient of the loss with respect to the input data,
creates a perturbed image by calling fgsm, then checks to see if the perturbed example is adversarial.
### Result

We can see the accuracy dropping :

![capture](https://user-images.githubusercontent.com/45148200/49972073-e2992300-ff30-11e8-97f3-d2ac764abddb.PNG)

And here are some examples of wrong predictions

![imshow](https://user-images.githubusercontent.com/45148200/49972138-fcd30100-ff30-11e8-90d1-d32fb81d4dc6.PNG)
