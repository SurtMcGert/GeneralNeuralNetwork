package NeuralNetwork;

public class ConvolutionalLayer implements Layer{

    // weights and biases
    private Matrix weights;
    private Matrix bias;
    private Matrix inputs;
    private Matrix rawOutput;
    private Matrix output;
    private Function activation;
    private Function activationDeriv;

    @Override
    public Matrix feedForward(Matrix inp) throws Exception {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Matrix backPropogate(Matrix errors, double lr) throws Exception {
        // TODO Auto-generated method stub
        return null;
    }
    
}
