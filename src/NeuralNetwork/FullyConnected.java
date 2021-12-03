package NeuralNetwork;

public class FullyConnected implements Layer {

  // weights and biases
  private Matrix weights;
  private Matrix bias;
  private Matrix rawOutput;
  private Function activation;

  /**
   * function to create a new fully connected layer
   * 
   * @param inputs  - the number of inputs to the layer
   * @param outputs - the number of outputs from the layer
   */
  public FullyConnected(int inputs, int outputs, Function activation) {
    this.weights = new Matrix(outputs, inputs);
    this.bias = new Matrix(outputs, 1);
    this.rawOutput = new Matrix(outputs, 0);
    this.activation = activation;
  }

  @Override
  public Matrix feedForward(Matrix inp) {
    try {
      this.rawOutput = Matrix.staticMultiply(this.weights, inp);
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return Matrix.staticMap(this.rawOutput, (double output) -> this.activation.function(output));
  }

  @Override
  public Matrix backPropogate(Matrix inp) {
    // TODO Auto-generated method stub
    return null;
  }

}
