package generalneuralnetwork;

import NeuralNetwork.FullyConnectedLayer;
import NeuralNetwork.Matrix;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.ActivationFunctions;
import NeuralNetwork.ConvolutionalLayer;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author harry
 */
public class GeneralNeuralNetwork {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */

    public GeneralNeuralNetwork() throws Exception {

        NeuralNetwork nn = new NeuralNetwork();
        // FullyConnectedLayer fc1 = new FullyConnectedLayer(2, 3, (double inp) ->
        // ActivationFunctions.sigmoid(inp),
        // (double inp) -> ActivationFunctions.sigmoidDeriv(inp));
        // FullyConnectedLayer fc2 = new FullyConnectedLayer(3, 1, (double inp) ->
        // ActivationFunctions.sigmoid(inp),
        // (double inp) -> ActivationFunctions.sigmoidDeriv(inp));

        ConvolutionalLayer cl = new ConvolutionalLayer(2, 1, 3, (double inp) -> ActivationFunctions.sigmoid(inp),
                (double inp) -> ActivationFunctions.sigmoidDeriv(inp));
        nn.addLayer(cl);

        double[][][] inp = {
                { { 1.4, 0.6, 0.1 }, { 0.3, 0.1, 1.7 }, { 1.3, 0.4, 0.2 } },
                { { 1.5, 0.7, 0.1 }, { 0.9, -0.1, 0.7 }, { 1.5, 0.1, 0.2 } } };
        // double[][][] inp = {
        // { { 1.4, 0.6, 0.1 }, { 0.3, 0.1, 1.7 }, { 1.3, 0.4, 0.2 } } };
        double[] output = nn.feedforward(inp);

        // nn.addLayer(fc1);
        // nn.addLayer(fc2);

        // double[][] input1 = { { 1.0, 1.0 } };
        // double[][] input2 = { { 1.0, 0.0 } };
        // double[][] input3 = { { 0.0, 1.0 } };
        // double[][] input4 = { { 0.0, 0.0 } };

        // double[] answer1 = { 1.0 };
        // double[] answer2 = { 0.0 };

        // for (int i = 0; i < 5000; i++) {
        // try {
        // nn.train(input1, answer2);
        // nn.train(input2, answer1);
        // nn.train(input3, answer1);
        // nn.train(input4, answer2);
        // } catch (Exception e) {
        // // TODO Auto-generated catch block
        // e.printStackTrace();
        // }

        // }

        // double[] output = { 0 };
        // try {
        // output = nn.feedforward(input2);
        // } catch (Exception e) {
        // // TODO Auto-generated catch block
        // e.printStackTrace();
        // }

        // for (int i = 0; i < output.length; i++) {
        // System.out.println(Math.round(output[i]));
        // }

    }

    public static void main(String[] args) throws Exception {
        new GeneralNeuralNetwork();
    }

}
