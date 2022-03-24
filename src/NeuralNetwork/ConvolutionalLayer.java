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
        int inputWidth = input[0].getColumns();
        int possibleFits = inputWidth - this.kernalSize + 1;
        // format the filters into a sparse matrix of weights
        Matrix formattedWeights = new Matrix(flattenedInputs.getColumns(),
                this.kernalSize * this.kernalSize * this.filters.getColumns());
        int moveDownCounter = 0;
        for (int column = 0; column < this.kernalSize * this.kernalSize; column++) {
            int filterIndex = 0;
            int fitCounter = 0;
            if (column % possibleFits == 0) {
                if (column != 0) {
                    moveDownCounter++;
                }
            }
            int rowMoveDownCounter = 0;
            for (int row = 0; row < input[0].getColumns() * input[0].getRows(); row++) {
                int filterRowOffset = 0;
                rowLoop: if ((fitCounter < possibleFits) && (filterIndex < this.kernalSize * this.kernalSize)) {
                    for (int rowOffset = 0; rowOffset < formattedWeights
                            .getRows(); rowOffset += input[0].getColumns()
                                    * input[0].getRows()) {
                        int filter = 0;
                        for (int columnOffset = 0; columnOffset < formattedWeights
                                .getColumns(); columnOffset += this.kernalSize * this.kernalSize) {
                            if (row < column) {
                                // formattedWeights.set(row + rowOffset, column + columnOffset, 0);
                                break rowLoop;
                            } else if (rowMoveDownCounter < moveDownCounter) {
                                // formattedWeights.set(row + rowOffset, column + columnOffset, 0);
                                rowMoveDownCounter++;
                                break rowLoop;
                            } else {
                                formattedWeights.set(row + rowOffset, column + columnOffset,
                                        this.filters.valAt(filterIndex + filterRowOffset, filter));
                            }
                            filter++;
                            rowMoveDownCounter++;
                        }
                        filterRowOffset += this.kernalSize * this.kernalSize;

                    }
                    filterIndex++;
                    fitCounter++;
                } else {
                    fitCounter = 0;
                    // for (int offset = 0; offset < formattedWeights.getRows(); offset +=
                    // input[0].getColumns()
                    // * input[0].getRows()) {
                    // formattedWeights.set(row + offset, column, 0);
                    // }
                }
            }
        }

        formattedWeights.view();
        Matrix multiplied = Matrix.staticMultiply(flattenedInputs, formattedWeights);

        return this.output;

        // this.inputs = input;
        // Matrix currentFeature;
        // int outputMatrixWidth = (input[0].getColumns() - this.kernalSize) + 1;
        // int outputMatrixHeight = (input[0].getRows() - this.kernalSize) + 1;
        // Matrix temp = new Matrix(outputMatrixHeight, outputMatrixWidth);
        // for (int i = 0; i < this.rawOutput.length; i++) {
        // this.rawOutput[i] = temp;
        // }

        // for (int y = 0; y < outputMatrixHeight; y++) {
        // for (int x = 0; x < outputMatrixWidth; x++) {
        // currentFeature = getConvolution(x, y);
        // Matrix currentConvolvedFeatures = Matrix.staticMultiply(currentFeature,
        // this.filters);
        // for (int i = 0; i < this.rawOutput.length; i++) {
        // this.rawOutput[i].set(y, x, currentConvolvedFeatures.valAt(0, i));
        // this.rawOutput[i].view();
        // }
        // }
        // }

        // for (int i = 0; i < this.output.length; i++) {
        // this.output[i] = Matrix.staticMap(this.rawOutput[i], (double output) ->
        // this.activation.function(output));
        // }

        // return this.output;
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

    /**
     * function to get a kernal sized area of each input streams and arrange them as
     * a row in a matrix
     * 
     * @param inpX - the x position of index to get the area from
     * @param inpY - the y position of the index to get the area from
     * @return - a kernal sized area of each input stream arranged as a row in a
     *         matrix
     * @throws Exception
     */
    private Matrix getConvolution(int inpX, int inpY) throws Exception {
        Matrix convolution;
        double[] output = new double[this.kernalSize * this.kernalSize * this.inputs.length];
        int pos = 0;

        for (int stream = 0; stream < this.inputs.length; stream++) {
            for (int x = inpX; x < inpX + this.kernalSize; x++) {
                for (int y = inpY; y < inpY + this.kernalSize; y++) {
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
