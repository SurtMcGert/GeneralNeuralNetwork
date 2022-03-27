package NeuralNetwork;

public class PoolingLayer implements Layer{
    private int kernalSize;
    public enum poolingTypes{MAX, MIN, AVERAGE}
    private poolingTypes type;

    /**
     * create a new pooling layer
     * @param kernalSize
     * @param type
     */
    public PoolingLayer(int kernalSize, poolingTypes type){
        this.kernalSize = kernalSize;
        this.type = type;
    }

    @Override
    public Matrix[] feedForward(Matrix[] input) throws Exception {
        switch(this.type){
            case MAX:
                break;
            case MIN:
                break;
            case AVERAGE:
                break;
            
        }
        return null;
    }

    @Override
    public Matrix[] backPropogate(Matrix[] errors, double lr) throws Exception {
        // TODO Auto-generated method stub
        return null;
    }
    
}
