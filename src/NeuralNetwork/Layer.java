package NeuralNetwork;

public interface Layer {

  /**
   * interface for allowing a user to input a activation function
   */
  interface Function {
    public double function(double a);
  }

  /**
   * function to feed data through the layer
   * 
   * @param inp - the list of inputs to pass through
   * @return double[] - the output of this layer
   */
  public Matrix feedForward(Matrix inp);

  /**
   * function to backPropogate through the layer
   * 
   * @param inp - the list of data to propogate backwards with
   * @return double[] - the new values of this layer
   */
  public Matrix backPropogate(Matrix inp);
}
