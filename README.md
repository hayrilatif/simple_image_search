# Simple Image Search
A simple autoencoder is trained, and its encoder part is used to create vector representations of images. When you want to search for an image in an index, the program compares the distance of the representation of your image to other representations of the images listed in the index. The autoencoder is trained on the CIFAR-10 dataset, so all images are resized to 32x32 before calculations. You can save and reuse the index file.

The dataset used to train the autoencoder model in this repository is:

@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
