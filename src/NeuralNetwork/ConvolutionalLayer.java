package NeuralNetwork;

public class ConvolutionalLayer implements Layer{

    // weights and biases
    private Matrix filters;
    private int kernalSize;
    private int numOfInputs;
    private Matrix bias;
    private Matrix[] inputs;
    private Matrix[] rawOutput;
    private Matrix[] output;
    private Function activation;
    private Function activationDeriv;


    /**
     * make a new convolution layer
     * @param inputChannels - the number of input channels to the network e.g 3 for an RGB image
     * @param outputs - the number of outputs this layer should have
     * @param kernalSize - the size of each kernal
     * @param activation - the activation function to apply to this layer
     * @param activationDeriv - the derivative of the given activation function
     * @throws Exception
     */
    public ConvolutionalLayer(int inputChannels, int outputs, int kernalSize, Function activation, Function activationDeriv) throws Exception {
        this.numOfInputs = inputChannels;
        this.filters = new Matrix(kernalSize * kernalSize * inputChannels, outputs);
        this.filters.randVal();
        if(kernalSize %2 != 0){
            this.kernalSize = kernalSize;
        }else{
            throw new Exception("kernal size must be odd");
        }
        this.rawOutput = new Matrix[outputs];
        this.output = new Matrix[outputs];
        this.activation = activation;
        this.activationDeriv = activationDeriv;
    }

    @Override
    public Matrix[] feedForward(Matrix[] input) throws Exception {
        this.inputs = input;
        Matrix currentFeature;
        int outputMatrixWidth = (input[0].getColumns() - this.kernalSize) + 1;
        int outputMatrixHeight = (input[0].getRows() - this.kernalSize) + 1;
        Matrix temp = new Matrix(outputMatrixHeight, outputMatrixWidth);
        for(int i = 0; i < this.rawOutput.length; i++){
            this.rawOutput[i] = temp;
        }

        for(int y = 0; y < outputMatrixHeight; y++){
            for(int x = 0; x < outputMatrixWidth; x++){
                currentFeature = getConvolution(x, y);
                Matrix currentConvolvedFeatures = Matrix.staticMultiply(currentFeature, this.filters);
                for(int i = 0; i < this.rawOutput.length; i++){
                    this.rawOutput[i].set(y, x, currentConvolvedFeatures.valAt(0, i));
                    this.rawOutput[i].view();
                }
            }
        }

        for(int i = 0; i < this.output.length; i++){
            this.output[i] = Matrix.staticMap(this.rawOutput[i], (double output) -> this.activation.function(output));
        }


        
        return this.output;
    }

    @Override
    public Matrix[] backPropogate(Matrix[] errors, double lr) throws Exception {
        //calculate the gradient
        Matrix[] gradient = errors;
        for(int i = 0; i < gradient.length; i++){
            gradient[i].hadamardProduct(Matrix.staticMap(this.output[i], (double inp) -> this.activationDeriv.function(inp)));
        }

        //calculate the errors in this layers inputs
        Matrix[] hiddenErrors = gradient;
        for(int i = 0; i < hiddenErrors.length; i++){
            hiddenErrors[i].multiply(this.filters.sum(i));
        }

        for(int i = 0; i < gradient.length; i++){
            gradient[i].multiply(lr);
        }

        //calculate the deltas
        

        
        return null;
    }

    /**
     * function to get a kernal sized area of each input streams and arrange them as a row in a matrix
     * @param inpX - the x position of index to get the area from
     * @param inpY - the y position of the index to get the area from
     * @return - a kernal sized area of each input stream arranged as a row in a matrix
     * @throws Exception
     */
    private Matrix getConvolution(int inpX, int inpY) throws Exception{
        Matrix convolution;
        double[] output = new double[this.kernalSize * this.kernalSize * this.inputs.length];
        int pos = 0;
        
        for(int stream = 0; stream < this.inputs.length; stream++){
            for(int x = inpX; x < inpX + this.kernalSize; x++){
                for(int y = inpY; y < inpY + this.kernalSize; y++){
                    output[pos] = this.inputs[stream].valAt(y, x);
                    pos++;
                }
            }
        }

        convolution = new Matrix(output);
        convolution.transpose();

        return convolution;
    }
    
}
