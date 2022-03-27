package NeuralNetwork;

public class ConvolutionalLayer implements Layer {

    // weights and biases
    private Matrix filters;
    private int kernalSize;
    private int numOfInputs;
    private Matrix bias;
    private Matrix inputs;
    private Matrix[] rawOutput;
    private Matrix[] output;
    private Function activation;
    private Function activationDeriv;

    /**
     * make a new convolution layer
     * 
     * @param inputChannels   - the number of input channels to the network e.g 3
     *                        for an RGB image
     * @param outputs         - the number of outputs this layer should have
     * @param kernalSize      - the size of each kernal
     * @param activation      - the activation function to apply to this layer
     * @param activationDeriv - the derivative of the given activation function
     * @throws Exception
     */
    public ConvolutionalLayer(int inputChannels, int outputs, int kernalSize, Function activation,
            Function activationDeriv) throws Exception {
        this.numOfInputs = inputChannels;
        this.filters = new Matrix(kernalSize * kernalSize * inputChannels, outputs);
        this.filters.randVal();
        this.kernalSize = kernalSize;
        this.rawOutput = new Matrix[outputs];
        this.output = new Matrix[outputs];
        this.activation = activation;
        this.activationDeriv = activationDeriv;
    }

    @Override
    public Matrix[] feedForward(Matrix[] input) throws Exception {
        // flatten the inputs
        Matrix flattenedInputs = flattenInputs(input);
        this.inputs = flattenedInputs;
        // format the filters into a sparse matrix of weights
        Matrix formattedWeights = convertWeightsToSparseMatrix(input);

        //get each output value
        Matrix multiplied = Matrix.staticMultiply(flattenedInputs, formattedWeights);

        //format the outupt into arrays of matrices
        int possibleHorizontalFits = input[0].getColumns() - this.kernalSize + 1;
        int possibleVerticalFits = input[0].getRows() - this.kernalSize + 1;
        int outputIndex = 0;
        for(int i = 0; i < this.rawOutput.length; i++){
            Matrix m = new Matrix(possibleVerticalFits, possibleHorizontalFits);
            for(int row = 0; row < possibleVerticalFits; row++){
                for(int column = 0; column < possibleHorizontalFits; column++){
                    m.set(row, column, multiplied.valAt(0, outputIndex));
                    outputIndex++;
                }
            }
            this.rawOutput[i] = m;
            //put output through activation function
            this.output[i] = Matrix.staticMap(this.rawOutput[i], (double inp) -> this.activation.function(inp));
        }
        return this.output;
    }

    @Override
    public Matrix[] backPropogate(Matrix[] errors, double lr) throws Exception {
        // calculate the gradient
        Matrix[] gradient = errors;
        for (int i = 0; i < gradient.length; i++) {
            gradient[i].hadamardProduct(
                    Matrix.staticMap(this.output[i], (double inp) -> this.activationDeriv.function(inp)));
        }

        // calculate the errors in this layers inputs
        Matrix[] hiddenErrors = gradient;
        for (int i = 0; i < hiddenErrors.length; i++) {
            hiddenErrors[i].multiply(this.filters.sum(i));
        }

        for (int i = 0; i < gradient.length; i++) {
            gradient[i].multiply(lr);
        }

        // calculate the deltas

        return null;
    }

    /**
     * function to convert the weights of the filters into a sparse matrix
     * 
     * @param input - the inputs to this layer
     * @return - a sparse matrix of all the weights
     * @throws Exception
     */
    private Matrix convertWeightsToSparseMatrix(Matrix[] input) throws Exception{
        int inputWidth = input[0].getColumns();
        int inputHeight = input[0].getRows();
        int possibleHorizontalFits = inputWidth - this.kernalSize + 1;
        int possibleVerticalFits = inputHeight - this.kernalSize + 1;
        Matrix formattedWeights;
        formattedWeights = new Matrix(this.inputs.getColumns(), possibleHorizontalFits*possibleVerticalFits * this.filters.getColumns());
        int padding = inputWidth - this.kernalSize;

        for(int column = 0; column < possibleHorizontalFits * possibleVerticalFits; column++){
            int row = column / possibleHorizontalFits;
            if(row > 0){
                row *= inputWidth;
            }
            row += (column % possibleHorizontalFits);
            int startPadding = row;
            int filterRow = 0;
            for(row = row; row < inputWidth * inputHeight; row++){
                for(int sparseColumn = 0; sparseColumn < formattedWeights.getColumns(); sparseColumn+= possibleHorizontalFits * possibleVerticalFits){
                    for(int sparseRow = 0; sparseRow < formattedWeights.getRows(); sparseRow+= inputWidth*inputHeight){
                        formattedWeights.set(row + sparseRow, column + sparseColumn, this.filters.valAt(filterRow, sparseColumn / (possibleHorizontalFits * possibleVerticalFits)));
                    }
                }
                if((filterRow + 1)== this.kernalSize * this.kernalSize){
                    break;
                }
                if((((row - startPadding) + 2) % inputWidth == 0) && (row != 0)){
                    row += padding;
                }
                filterRow++;
            }
        }
        return formattedWeights;
    }

    /**
     * function to take an array of matrices and flatten them into a row of a matrix
     * 
     * @param inputs - the matrices to flatten
     * @return - the flattened inputs
     * @throws Exception
     */
    private Matrix flattenInputs(Matrix[] inputs) throws Exception {
        int rows = inputs.length;
        int columns = inputs[0].getRows() * inputs[0].getColumns();
        Matrix flattenedInputs = new Matrix(1, columns * inputs.length);

        int flattenedColumn = 0;
        for (int row = 0; row < inputs[0].getRows(); row++) {
            for (int column = 0; column < inputs[0].getColumns(); column++) {
                int offset = 0;
                for (int stream = 0; stream < inputs.length; stream++) {
                    flattenedInputs.set(0, flattenedColumn + offset, inputs[stream].valAt(row, column));
                    offset += columns;
                }
                flattenedColumn++;
            }
        }

        return flattenedInputs;
    }

    // /**
    // * function to get a kernal sized area of each input streams and arrange them
    // as
    // * a row in a matrix
    // *
    // * @param inpX - the x position of index to get the area from
    // * @param inpY - the y position of the index to get the area from
    // * @return - a kernal sized area of each input stream arranged as a row in a
    // * matrix
    // * @throws Exception
    // */
    // private Matrix getConvolution(int inpX, int inpY) throws Exception {
    // Matrix convolution;
    // double[] output = new double[this.kernalSize * this.kernalSize *
    // this.inputs.length];
    // int pos = 0;

    // for (int stream = 0; stream < this.inputs.length; stream++) {
    // for (int x = inpX; x < inpX + this.kernalSize; x++) {
    // for (int y = inpY; y < inpY + this.kernalSize; y++) {
    // output[pos] = this.inputs[stream].valAt(y, x);
    // pos++;
    // }
    // }
    // }

    // convolution = new Matrix(output);
    // convolution.transpose();

    // return convolution;
    // }

}
