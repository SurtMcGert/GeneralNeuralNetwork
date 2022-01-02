package NeuralNetwork;

public class ConvolutionalLayer implements Layer{

    // weights and biases
    private Matrix[] convolutions;
    private int inputWidth;
    private int convolutionStride;
    private int numOfInputs;
    private Matrix bias;
    private Matrix inputs;
    private Matrix rawOutput;
    private Matrix output;
    private Function activation;
    private Function activationDeriv;

    public ConvolutionalLayer(int inputs, int input2DArrayWidth, int convolutions, int convolutionStride, Function activation, Function activationDeriv) throws Exception {
        this.numOfInputs = inputs;
        this.inputWidth = input2DArrayWidth;
        this.convolutions = new Matrix[convolutions];
        Matrix temp;
        for(int i = 0; i < convolutions; i++){
            temp = new Matrix(convolutionStride, convolutionStride);
            temp.randVal();
            this.convolutions[i] = temp;
        }
        if(convolutionStride %2 != 0){
            this.convolutionStride = convolutionStride;
        }else{
            throw new Exception("convolution stride must be odd");
        }
        this.activation = activation;
        this.activationDeriv = activationDeriv;
    }

    @Override
    public Matrix feedForward(Matrix inp) throws Exception {
        this.inputs = inp;
        Matrix output = new Matrix((inp.getRows() * this.convolutions.length), 1);
        Matrix currentConvolution = new Matrix(this.convolutionStride, this.convolutionStride);

        for(int i = 0; i < inp.getRows(); i++){
            currentConvolution = new Matrix(getConvolution(i));
            for(int j = 0; j < this.convolutions.length; j++){
                Matrix temp = Matrix.staticHadamardProduct(currentConvolution, this.convolutions[j]);
                double sum = temp.sum();
                sum = sum / Math.pow(this.convolutionStride, 2);
                output.set((i + (this.numOfInputs * j)), 0, sum);
            }
        }
        this.rawOutput = output;
        output = Matrix.staticMap(this.rawOutput, (double input) -> this.activation.function(input));
        this.output = output;
        output.view();
        return output;
    }

    @Override
    public Matrix backPropogate(Matrix errors, double lr) throws Exception {
        // TODO Auto-generated method stub
        Matrix hiddenErrors = new Matrix(this.numOfInputs, 1);
        Matrix currentConvolution = new Matrix(this.convolutionStride, this.convolutionStride);
        for(int i = 0; i < errors.getRows(); i++){
            currentConvolution = new Matrix(getConvolution(i));
            for(int j = 0; j < this.convolutions.length; j++){

            }
        }
        return null;
    }

    /**
     * function to make a matrix of all the values in the input centerd around the given value.
     * The input to the convolutional layer is a 2D array/matrix which has been made 1D and the input to this function is the index in that 1D array that the desired point is at
     * @param inp - the value to center the the convolution around
     * @return - a matrix containing all the values from the input to this layer, centerd around the given value
     * @throws Exception
     */
    private Matrix getConvolution(int inp) throws Exception{
        Matrix convolution = new Matrix(this.convolutionStride, this.convolutionStride);
        int halfStride = (int)Math.floor(this.convolutionStride / 2);

        int posOfCenter = inp;
        int valuePosition = 0;
        double value = 0;


        for(int yOffset = -halfStride; yOffset <= halfStride; yOffset++){
            for(int xOffset = -halfStride; xOffset <= halfStride; xOffset++){
                valuePosition = (posOfCenter + xOffset) + (yOffset * this.inputWidth);
                if((valuePosition >= 0) && (valuePosition < (this.inputWidth * (yOffset + (posOfCenter / this.inputWidth) + 1))) && (valuePosition > (((posOfCenter / this.inputWidth) * this.inputWidth) + (this.inputWidth * yOffset) - 1)) && (valuePosition < this.inputs.getRows())){
                    value = this.inputs.valAt(valuePosition, 0);
                }else{
                    value = 0;
                }
                convolution.set(halfStride + yOffset, halfStride + xOffset, value);
            }
        }
        return convolution;
    }

    /**
     * function to return the number of outputs this layer will return
     * @return int
     */
    public int getNumOfOutputs(){
        return this.numOfInputs * this.convolutions.length;
    }
    
}
