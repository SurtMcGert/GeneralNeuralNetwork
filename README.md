# GeneralNeuralNetwork
this is a simple dynamic fully connected neural network library for java
the library allowes for the following:
- create a fully connected neural network with any number of inputs, hidden layers, hidden nodes and outputs
- save networks to binary files
- read networks from binary files that you previously saved
- train the network
- test the network


download the code and put the "NeuralNetwork" folder inside your projects "src" folder
you will then be able to import the neural network library into your projects other classes

-------how to use--------

NeuralNetwork nn =  new NeuralNetwork({1,2,3,4});
this will create a new neural network with:
1 input
2 nodes in the first hidden layer
3 nodes in the second hidden layer
4 outputs

nn.saveNetwork("firstNetwork")
will save all the values of the network in a binary file called "firstNetwork"

nn.loadNetwork("firstNetwork")
will load the network you just saved to the binary file called "firstNetwork"

nn.train(<array of inputs>, <array of expectations>)
will train the network by inputting the first array and comparing the networks output to the second array (what the answer is)

nn.feedfoward(<array of inputs>)
will pass your array of inputs through the network and return the networks output as an array
