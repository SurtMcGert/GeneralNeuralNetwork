package NeuralNetwork;

public class FullyConnectedLayer implements Layer {

  // weights and biases
  private Matrix weights;
  private Matrix bias;
  private Matrix inputs;
  private Matrix rawOutput;
  private Matrix output;
  private Function activation;
  private Function activationDeriv;

  /**
   * function to create a new fully connected layer
   * 
   * @param inputs  - the number of inputs to the layer
   * @param outputs - the number of outputs from the layer
   */
  public FullyConnectedLayer(int inputs, int outputs, Function activation, Function activationDeriv) {
    try {
      this.weights = new Matrix(outputs, inputs);
      this.weights.randVal();
      this.bias = new Matrix(outputs, 1);
      this.bias.randVal();
      this.rawOutput = new Matrix(outputs, 1);
      this.activation = activation;
      this.activationDeriv = activationDeriv;
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    
  }

  @Override
  public Matrix feedForward(Matrix inp) throws Exception {
    if(inp.getRows() == this.weights.getColumns()){
      this.inputs = inp;
      try {
        this.rawOutput = Matrix.staticMultiply(this.weights, inp);
        this.rawOutput.add(this.bias);
      } catch (Exception e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
      this.output = Matrix.staticMap(this.rawOutput, (double output) -> this.activation.function(output));
    }else{
      throw new Exception("incorrect number of inputs, should be " + this.weights.getColumns());
    }
    return this.output;
  }

  @Override
  public Matrix backPropogate(Matrix errors, double lr) throws Exception {
    //calculate the errors in this layers inputs
    Matrix tw = Matrix.staticTranspose(this.weights);
    Matrix hiddenErrors = Matrix.staticMultiply(tw, errors);

    //calculate the gradient
    Matrix gradient = Matrix.staticMap(this.output, (double inp) -> this.activationDeriv.function(inp));
    gradient.hadamardProduct(errors);
    gradient.multiply(lr);

    //calculate the deltas
    Matrix transposedHiddens = Matrix.staticTranspose(this.inputs);
    Matrix weightDeltas = Matrix.staticMultiply(gradient, transposedHiddens);

    //add the deltas to the weights
    this.weights.add(weightDeltas);
    this.bias.add(gradient);

    return hiddenErrors;
  }

}
