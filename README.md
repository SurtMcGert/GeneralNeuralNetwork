# GNN
GNN is a simple dynamic neural network library for java

# installation
just download the repo and copy the "NeuralNetwork" package into your project

# Usage
```java
NeuralNetwork nn = new NeuralNetwork();
```
creates a new neural network

```java
FullyConnectedLayer fc1 = new FullyConnectedLayer(<num of inputs>, <num of outputs>, <activation function>, <activation function derivative>);
```
creates a new fully connected layer

```java
ConvolutionalLayer cl = new ConvolutionalLayer(<num of input channels>, <num of outputs>, <kernal size>, <activation function>, <activation function derivative>);
```
creates a new convolutional layer

```java
nn.addLayer(<layer name>);
```
adds a new layer to the neural network

```java
nn.feedforward(inp);
```
feeds data through the network and retunrs the output

```java
nn.train(<input>, <expectation>);
```
trains the neural network on the input and the expectations

# in the future
add some way to save and load networks<br />
add a way to load a network from a json file<br />
add full functionality for the convolutional layer<br />
add pooling layer<br />


# Contributing
... please dont lol

