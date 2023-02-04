# Neural-Network-for-Image-OCR

The attached files contains the program written in Python consisting a generalized neuralnetwork class and a program file consisting a NN network model trained on MNIST dataset.

The neuralnetwork class takes the number of layers and shape from the user to develop a network as numpy matrices. It assigns random weights to each nodes at the start and those weights gets updated as the model is trained on the dataset. For training, we feed forward the data through the model, and then based on the error at output layer, we move backwards through the model and update the weight values based on the respective error value. This model only uses sigmoid function for the coumputations, but it can also be done using other computational functions, such as relu, etc. More inforamtion on the mechanism is given in the code with comments.
