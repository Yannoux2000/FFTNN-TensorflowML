# FFTNN-TensorflowML
CNN made out of fft, for classification tasks.
This Classification Neural Network is made with tensorflow, make sure you have it installed, before you try this script.
I got inspired from a class i had, where the teacher told us that convolution in the time domain, was equivalent to a multiplication in a frequancy domain. i tried it and compared it with the CNN from Tensorflow's examples CNN. i stared this repo if you want to check it out, aymericdamien/TensorFlow-Examples.

# Applied to the MNIST

i have reached accuracy of about 99% with CNN, where as with the fftnn i reached around 95%
the main interest for the FFTNN is that it offers way better performances concerning fps.
5 batches of 128 on CNN, against 60 batches on FFTNN

check this out on your setup ;)

# Details and thoughs on this

This algorithm is just like a convolution (the fftnn) it is concidered that converting a convolution in time domain, into a multiplication in an transformed domain, would be equivalent. and it is. This algorithm is a proof, if you were still skeptical, like i was. the main advantages from this algorithm are dense, it can be applied to a 2d image since images have a determined width. this makes a dot on a drawing periodic. It can adapt from slightly different sizes (i runned this algorithm with an fftlength shorter than the images one, and it stills perform pretty good. i have to check one thing about reshaping an array in tensorflow. this might be the reason why fft was faster than cnn, because of resize time. which i realized recently can be really slow in some computers, like mine.
