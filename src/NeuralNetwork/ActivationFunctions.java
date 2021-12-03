package NeuralNetwork;

final class ActivationFunctions {
  private ActivationFunctions() throws Exception {
    throw new Exception("Cannot be instantiated");
  }

  /**
   * the sigmoid function
   * 
   * @param input - the value to pass through
   * @return double
   */
  public static double sigmoid(double input) {
    return 1.0 / (1.0 + Math.exp(-input));
  }

  /**
   * the derivative of the sigmoid function
   * 
   * @param input - the value to pass through
   * @return double
   */
  public static double digmoidDeriv(double input) {
    return 0.0;
  }
}
