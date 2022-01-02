package NeuralNetwork;

public final class ActivationFunctions {
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
  // TODO - implement
  public static double sigmoidDeriv(double input) {
    return input * (1 - input);
  }

  /**
   * the relu function
   * @param input - the value to pass through
   * @return double
   */
  public static double relu(double input){
    return Math.max(0, input);
  }

  /**
   * the derivative of the relu function
   * @param input - the value to pass through
   * @return double
   */
  public static double reluDeriv(double input){
    double output = 0;
      if (input > 0) {
        output = 1;
      } else {
        output = 0;
      }
      return output;
  }
}
