package NeuralNetwork;

public class ConvolutionalLayer implements Layer {

    // weights and biases
    private Matrix filters;
    private Matrix formattedWeights;
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
        Matrix flattenedInputs = flattenMatrixArray(input);
        this.inputs = input;
        // format the filters into a sparse matrix of weights
        Matrix formattedWeights = convertWeightsToSparseMatrix(input, this.filters);
        this.formattedWeights = formattedWeights;

        // get each output value
        Matrix multiplied = Matrix.staticMultiply(flattenedInputs, formattedWeights);

        // format the outupt into arrays of matrices
        int possibleHorizontalFits = input[0].getColumns() - this.kernalSize + 1;
        int possibleVerticalFits = input[0].getRows() - this.kernalSize + 1;
        int outputIndex = 0;
        for (int i = 0; i < this.rawOutput.length; i++) {
            Matrix m = new Matrix(possibleVerticalFits, possibleHorizontalFits);
            for (int row = 0; row < possibleVerticalFits; row++) {
                for (int column = 0; column < possibleHorizontalFits; column++) {
                    m.set(row, column, multiplied.valAt(0, outputIndex));
                    outputIndex++;
                }
            }
            this.rawOutput[i] = m;
            // put output through activation function
            this.output[i] = Matrix.staticMap(this.rawOutput[i], (double inp) -> this.activation.function(inp));
        }
        return this.output;
    }

    @Override
    public Matrix[] backPropogate(Matrix[] errors, double lr) throws Exception {
        // calculate the error in this layers input
        Matrix flattenedError = this.flattenMatrixArray(errors);
        Matrix he = Matrix.staticMultiply(this.formattedWeights, Matrix.staticTranspose(flattenedError));
        he.transpose();
        // format the hidden errors into arrays of matrices
        Matrix[] hiddenErrors = this.inputs;
        int counter = 0;
        for (int row = 0; row < this.inputs[0].getRows(); row++) {
            for (int column = 0; column < this.inputs[0].getColumns(); column++) {
                for (int i = 0; i < this.numOfInputs; i++) {
                    hiddenErrors[i].set(row, column,
                            he.valAt(0, counter + ((this.inputs[0].getRows() * this.inputs[0].getColumns()) * i)));
                }
                counter++;
            }
        }
        // calculate the error in this layers weights
        // dc/dw = dc/da * da/dz * dz/dw
        // da/dz
        Matrix da_dz = Matrix.staticMap(this.flattenMatrixArray(this.rawOutput),
                (double inp) -> this.activationDeriv.function(inp));

        Matrix flattenedInputs = this.flattenMatrixArray(this.inputs);
        // get the deltas
        Matrix weightDeltas = Matrix.staticMultiply(da_dz, lr);
        weightDeltas.hadamardProduct(flattenedError);

        // turn weightDeltas into a sparse matrix
        Matrix sparseWeightMatrix = new Matrix(weightDeltas.getColumns() / this.output.length, this.output.length);
        counter = 0;
        for (int row = 0; row < sparseWeightMatrix.getRows(); row++) {
            for (int column = 0; column < sparseWeightMatrix.getColumns(); column++) {
                sparseWeightMatrix.set(row, column, weightDeltas.valAt(0, counter));
                counter++;
            }
        }
        sparseWeightMatrix = this.convertWeightsToSparseMatrix(this.inputs, sparseWeightMatrix);
        weightDeltas = Matrix.staticMultiply(flattenedInputs, sparseWeightMatrix);
        Matrix temp = weightDeltas;
        weightDeltas = new Matrix(weightDeltas.getColumns() / this.output.length, this.output.length);
        counter = 0;
        for (int row = 0; row < weightDeltas.getRows(); row++) {
            for (int column = 0; column < weightDeltas.getColumns(); column++) {
                weightDeltas.set(row, column, temp.valAt(0, counter));
                counter++;
            }
        }

        // add the deltas to train the weights
        this.filters.add(weightDeltas);
        // TODO - CALCULATE BIAS DELTAS

        return hiddenErrors;
    }

    /**
     * function to convert the weights of the filters into a sparse matrix
     * 
     * @param input - the inputs to this layer
     * @return - a sparse matrix of all the weights
     * @throws Exception
     */
    private Matrix convertWeightsToSparseMatrix(Matrix[] input, Matrix matrixToFormat) throws Exception {
        int inputWidth = input[0].getColumns();
        int inputHeight = input[0].getRows();
        int possibleHorizontalFits = inputWidth - this.kernalSize + 1;
        int possibleVerticalFits = inputHeight - this.kernalSize + 1;
        Matrix formattedWeights;
        formattedWeights = new Matrix((this.inputs[0].getColumns() * this.inputs[0].getRows()) * this.inputs.length,
                possibleHorizontalFits * possibleVerticalFits * matrixToFormat.getColumns());
        int padding = inputWidth - this.kernalSize;

        for (int column = 0; column < possibleHorizontalFits * possibleVerticalFits; column++) {
            int row = column / possibleHorizontalFits;
            if (row > 0) {
                row *= inputWidth;
            }
            row += (column % possibleHorizontalFits);
            int startPadding = row;
            int filterRow = 0;
            for (row = row; row < inputWidth * inputHeight; row++) {
                for (int sparseColumn = 0; sparseColumn < formattedWeights
                        .getColumns(); sparseColumn += possibleHorizontalFits * possibleVerticalFits) {
                    for (int sparseRow = 0; sparseRow < formattedWeights.getRows(); sparseRow += inputWidth
                            * inputHeight) {
                        formattedWeights.set(row + sparseRow, column + sparseColumn, matrixToFormat.valAt(filterRow,
                                sparseColumn / (possibleHorizontalFits * possibleVerticalFits)));
                    }
                }
                if ((filterRow + 1) == this.kernalSize * this.kernalSize) {
                    break;
                }
                if ((((row - startPadding) + 2) % inputWidth == 0) && (row != 0)) {
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
    private Matrix flattenMatrixArray(Matrix[] inputs) throws Exception {
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
}
