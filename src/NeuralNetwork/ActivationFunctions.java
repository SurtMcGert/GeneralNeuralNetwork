package NeuralNetwork;

public final class ActivationFunctions {
  private ActivationFunctions() throws Exception {
    throw new Exception("Cannot be instantiated");
  }

  /**
   * adaptive sigmoid function
   * @param beta - the beta parameter for the function
   * @param input - the value to pass through the function
   * @return double - the output of the sigmoid function with the given beta at the given input
   */
  public static double adaptSigmoid(double beta, double input){
    return 1.0 / (1.0 + Math.exp(-beta * input));
  }

  /**
   * the derivative of the adaptive sigmoid function
   * @param beta - the beta parameter for the sigmoid function
   * @param input - the value to pass through the function
   * @return double - the result of the derivative of the adaptive sigmoid function at the given input
   */
  public static double adaptSigmoidDeriv(double beta, double input){
    return (beta * input) * (1 - (beta * input));
  }

  /**
   * the sigmoid function
   * 
   * @param input - the value to pass through
   * @return double
   */
  public static double sigmoid(double input) {
    return adaptSigmoid(1, input);
  }

  /**
   * the derivative of the sigmoid function
   * 
   * @param input - the value to pass through
   * @return double
   */
  public static double sigmoidDeriv(double input) {
    return adaptSigmoidDeriv(1, input);
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

  /**
   * the tanh activation function
   * @param input - the value to pass in
   * @return double - the output of the tanh function at the given input
   */
  public static double tanh(double input){
    return (2 * ActivationFunctions.sigmoid(2 * input)) - 1;
  }

  /**
   * the derivative of the tanh function
   * @param input - the value to pass through
   * @return double - the output of the derivative of the tanh function at the given input
   */
  public static double tanhDeriv(double input){
    return 1 - (Math.pow(ActivationFunctions.tanh(input), 2));
  }

  /**
   * parameterised ReLu function
   * @param param - the value of the gradient of the line for x <= 0
   * @param input - the value to pass through the function
   * @return double - the output of the function at the given input
   */
  public static double parameterisedRelu(double param, double input){
    if(input > 0){
      return input;
    }else{
      return param * input;
    }
  }

  /**
   * the derivative of the parameterised ReLu function
   * @param param - the value of the gradient of the parametised ReLu function for x <= 0
   * @param input - the value to pass through the function
   * @return double - the output of the derivative of the parameterised ReLu function with the given param at the given input
   */
  public static double parameterisedReluDeriv(double param, double input){
    if(input > 0){
      return 1;
    } else{
      return param;
    }
  }

  /**
   * the leaky ReLu function
   * @param input - the value to pass through the function
   * @return double - the output of the function at the given input
   */
  public static double leakyRelu(double input){
    if(input > 0){
      return input;
    }else{
      return ActivationFunctions.parameterisedRelu(0.01, input);
    }
  }

  /**
   * the derivative of the leaky ReLu function
   * @param input - the value to pass through
   * @return double - the output of the derivative of the leaky ReLu function at the given input
   */
  public static double leakyReluDeriv(double input){
    if(input > 0){
      return 1;
    }else{
      return ActivationFunctions.parameterisedReluDeriv(0.01, input);
    }
  }

  /**
   * exponential linear unit function
   * @param a - the multiplier for the exopnential line for x <= 0
   * @param input - the value to pass through the function
   * @return double the output of the function
   */
  public static double elu(double a, double input){
    if(input > 0){
      return input;
    }else{
      return a * (Math.pow(Math.E, input) - 1);
    }
  }

  /**
   * derivative of the exponential inear unit function
   * @param a - the multiplier for the exponential line of the elu function for x <= 0
   * @param input - the value to pass through the function
   * @return double - the output of the derivative of the elu function with the given a at the given input
   */
  public static double eluDeriv(double a, double input){
    if(input > 0){
      return 1;
    }else{
      return a * (Math.pow(Math.E, input));
    }
  }

  /**
   * the swish activation function
   * @param beta - the beta parameter of the function
   * @param input - the value to pass through the function
   * @return double - the output of the function
   */
  public static double swish (double beta, double input){
    return input * ActivationFunctions.sigmoid(beta * input);
  }

  /**
   * the derivative of the swish function
   * @param beta - the beta parameter of the swish function
   * @param input - the value to pass through the network
   * @return - the output of the derivative of the swish function with the given beta value at the given input
   */
  public static double swishDeriv(double beta, double input){
    return (beta * ActivationFunctions.swish(beta, input)) + (ActivationFunctions.adaptSigmoid(beta, input) * (1 - (beta * ActivationFunctions.swish(beta, input))));
  }

  /**
   * the softmax function
   * @param input - the input to pass through the function
   * @return double[] - the output of softmax function at the given inputs
   */
  public static double[] softMax (double[] input){
    double[] output = new double[input.length];
    Matrix inputMatrix = new Matrix(input);
    double sum = inputMatrix.sum();
    for(int i = 0; i < output.length; i++){
      output[i] = input[i] / sum;
    }
    return output;
  }


  /**
   * the derivative of the softmax function
   * @param input - the input to pass through the derivative of the softmax function
   * @return double[] - the output of the derivative of the softmax function at the given inputs
   */
  public static double[] softmaxDeriv(double[] input){
    double[] output = new double[input.length];
    for(int i = 0; i < output.length; i++){
      output[i] = input[i] * (1 - input[i]);
    }
    return output;
  }


}
