# FFTNN-TensorflowML
CNN made out of fft, for classification tasks.
This Classification Neural Network is made with tensorflow, make sure you have it installed, before you try this script.
I got inspired from a class i had, where the teacher told us that convolution in the time domain, was equivalent to a multiplication in a frequancy domain. i tried it and compared it with the CNN from Tensorflow's examples CNN. i stared this repo if you want to check it out, aymericdamien/TensorFlow-Examples.

# Applied to the MNIST

i have reached accuracy of about 99% with CNN, where as with the fftnn i reached around 95%
the main interest for the FFTNN is that it offers way better performances concerning fps.
5 batches of 128 on CNN, against 60 batches on FFTNN

check this out on your setup ;)
