package network;

import java.util.List;

import data.Image;
import data.ImageDataSet;

import util.Log;


public class ConvolutionalNeuralNetwork {
    //this is the loss function for the output of the neural network
    LossFunction lossFunction;

    //this is the total number of weights in the neural network
    int numberWeights;

    //specifies if the CNN will use dropout
    boolean useDropout;
    //the dropout for nodes in the input layer
    double inputDropoutRate;
    //the dropout for nodes in the hidden layer
    double hiddenDropoutRate;

    //specify if the CNN will use batch normalization
    boolean useBatchNormalization;

    //the alpha value used to calculate the running
    //averages for batch normalization
    double alpha;

    //layers contains all the nodes in the neural network
    ConvolutionalNode[][] layers;

    public void createSmallNoPool(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {
        layers = new ConvolutionalNode[5][];

        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(0, 0, NodeType.INPUT, activationType /*doesn't matter for input*/, inputPadding /*padding*/ , batchSize, inputChannels, inputY, inputX, useDropout, inputDropoutRate, false, 0.0);
        layers[0][0] = inputNode;


        //first hidden layer has 4 nodes
        layers[1] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(1, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 20, 20, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[1][i] = node;

            numberWeights += node.getNumberWeights();

            new ConvolutionalEdge(layers[0][0], node, inputChannels /* will be 1 for MNIST or 3 for CIFAR-10 to get the other feature maps to only have 1 channel*/, 13, 13);

            numberWeights += inputChannels * 13 * 13;
        }


        //second hidden layer also has 4 nodes, because it's just the max pooling from the first
        layers[2] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(2, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 10, 10, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[2][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[1][j], node, 1, 11, 11); //11x11 to get down to 10x10

                numberWeights += 11 * 11;
            }
        }


        //third hidden layer is dense with 10 nodes
        layers[3] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            ConvolutionalNode node = new ConvolutionalNode(3, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[3][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[2][j], node, 1, 10, 10);

                numberWeights += 10 * 10;
            }
        }

        //output layer is dense with 10 nodes
        layers[4] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            //no dropout or BN on output nodes
            ConvolutionalNode node = new ConvolutionalNode(4, i, NodeType.OUTPUT, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, false, 0.0, false, 0.0); 
            layers[4][i] = node;

            //no bias on output nodes

            for (int j = 0; j < 10; j++) {
                new ConvolutionalEdge(layers[3][j], node, 1, 1, 1);

                numberWeights += 1;
            }
        }
    }

    public void createSmall(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {
        layers = new ConvolutionalNode[5][];

        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(0, 0, NodeType.INPUT, activationType /*doesn't matter for input*/, inputPadding /*padding*/ , batchSize, inputChannels, inputY, inputX, useDropout, inputDropoutRate, false, 0.0);
        layers[0][0] = inputNode;


        //first hidden layer has 4 nodes
        layers[1] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(1, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 20, 20, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[1][i] = node;

            numberWeights += node.getNumberWeights();

            new ConvolutionalEdge(layers[0][0], node, inputChannels /* will be 1 for MNIST or 3 for CIFAR-10 to get the other feature maps to only have 1 channel*/, 13, 13);

            numberWeights += inputChannels * 13 * 13;
        }


        //second hidden layer also has 4 nodes, because it's just the max pooling from the first
        layers[2] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(2, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 10, 10, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[2][i] = node;

            numberWeights += node.getNumberWeights();

            new PoolingEdge(layers[1][i], node, 2, 2); //stride of 2 and pool size of 2 for this max pooling operation
        }


        //third hidden layer is dense with 10 nodes
        layers[3] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            ConvolutionalNode node = new ConvolutionalNode(3, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[3][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[2][j], node, 1, 10, 10);

                numberWeights += 10 * 10;
            }
        }

        //output layer is dense with 10 nodes
        layers[4] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            //no dropout or BN on output nodes
            ConvolutionalNode node = new ConvolutionalNode(4, i, NodeType.OUTPUT, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, false, 0.0, false, 0.0);
            layers[4][i] = node;

            //no bias on output nodes

            for (int j = 0; j < 10; j++) {
                new ConvolutionalEdge(layers[3][j], node, 1, 1, 1);

                numberWeights += 1;
            }
        }
    }

    public void createLeNet5(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {
        //TODO: Programming Assignment 3 - Part 1: Implement creating a LeNet-5 CNN
        //make sure dropout is turned off on the last hidden layer
        layers = new ConvolutionalNode[8][];
        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(0, 0, NodeType.INPUT, activationType /*doesn't matter for input*/, inputPadding /*padding*/ , batchSize, inputChannels, inputY, inputX, useDropout, inputDropoutRate, false, 0.0);
        layers[0][0] = inputNode;
        layers[1] = new ConvolutionalNode[6];
        int i = 0;
        while(i < 6){
            ConvolutionalNode node = new ConvolutionalNode(1 /*layer*/, i /*number*/, NodeType.HIDDEN, activationType, 0 /*padding*/, batchSize, 1, 28, 28, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[1][i] = node;
            numberWeights += node.getNumberWeights();
            new ConvolutionalEdge(layers[0][0], node, inputChannels, 5, 5);
            numberWeights += inputChannels*5*5;
            i++;
        }
        layers[2] = new ConvolutionalNode[6];
        i = 0;
        while(i < 6){
            ConvolutionalNode node = new ConvolutionalNode(2 /*layer*/, i /*number*/, NodeType.HIDDEN, activationType, 0 /*padding*/, batchSize, 1, 14, 14, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[2][i] = node;
            numberWeights += node.getNumberWeights();
            new PoolingEdge(layers[1][i], node, 2, 2);
            i++;
        }
        layers[3] = new ConvolutionalNode[16];
        i = 0;
        while(i < 16){
            ConvolutionalNode node = new ConvolutionalNode(3 /*layer*/, i /*number*/, NodeType.HIDDEN, activationType, 0 /*padding*/, batchSize, 1, 10, 10, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[3][i] = node;
            numberWeights += node.getNumberWeights();
            if(i == 0){
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                numberWeights += 3*5*5;
            } else if (i == 1){
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                numberWeights += 3*5*5;
            } else if (i == 2){
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                numberWeights += 3*5*5;
            } else if (i == 3){
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                numberWeights += 3*5*5;
            } else if (i == 4){
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                numberWeights += 3*5*5;
            } else if (i == 5){
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                numberWeights += 3*5*5;
            } else if (i == 6){
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 7){
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 8){
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 9){
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 10){
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 11){
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 12){
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 13){
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 14){
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                numberWeights += 4*5*5;
            } else if (i == 15){
                new ConvolutionalEdge(layers[2][0], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][1], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][2], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][3], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][4], node, 1, 5, 5);
                new ConvolutionalEdge(layers[2][5], node, 1, 5, 5);
                numberWeights += 6*5*5;
            }
            i++;
        }
        layers[4] = new ConvolutionalNode[16];
        i = 0;
        while(i < 16){
            ConvolutionalNode node = new ConvolutionalNode(4 /*layer*/, i /*number*/, NodeType.HIDDEN, activationType, 0 /*padding*/, batchSize, 1, 5, 5, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[4][i] = node;
            numberWeights += node.getNumberWeights();
            new PoolingEdge(layers[3][i], node, 2, 2);
            i++;
        }
        layers[5] = new ConvolutionalNode[120];
        i = 0;
        while(i < 120){
            ConvolutionalNode node = new ConvolutionalNode(5 /*layer*/, i /*number*/, NodeType.HIDDEN, activationType, 0 /*padding*/, batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[5][i] = node;
            numberWeights += node.getNumberWeights();
            for(int j = 0; j < 16; j++){
                new ConvolutionalEdge(layers[4][j], node, 1, 5, 5);
                numberWeights += 5*5;
            }
            i++;
        }
        layers[6] = new ConvolutionalNode[84];
        i = 0;
        while(i < 84){
            ConvolutionalNode node = new ConvolutionalNode(6 /*layer*/, i /*number*/, NodeType.HIDDEN, activationType, 0 /*padding*/, batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[6][i] = node;
            numberWeights += node.getNumberWeights();
            for(int j = 0; j < 120; j++){
                new ConvolutionalEdge(layers[5][j], node, 1, 1, 1);
                numberWeights += 1;
            }
            i++;
        }
        layers[7] = new ConvolutionalNode[10];
        i = 0;
        while(i < 10){
            ConvolutionalNode node = new ConvolutionalNode(6 /*layer*/, i /*number*/, NodeType.OUTPUT, activationType, 0 /*padding*/, batchSize, 1, 1, 1, false, 0.0, false, 0.0);
            layers[7][i] = node;
            for(int j = 0; j < 84; j++){
                new ConvolutionalEdge(layers[6][j], node, 1, 1, 1);
                numberWeights += 1;
            }
            i++;
        }
    }

    public ConvolutionalNeuralNetwork(LossFunction lossFunction, boolean useDropout, double inputDropoutRate, double hiddenDropoutRate, boolean useBatchNormalization, double alpha) {
        this.lossFunction = lossFunction;
        this.useDropout = useDropout;
        this.inputDropoutRate = inputDropoutRate;
        this.hiddenDropoutRate = hiddenDropoutRate;
        this.useBatchNormalization = useBatchNormalization;
        this.alpha = alpha;
    }

    /**
     * This gets the number of weights in the ConvolutionalNeuralNetwork, which should
     * be equal to the number of hidden nodes (1 bias per hidden node) plus 
     * the number of edges (1 bias per edge). It is updated whenever an edge 
     * is added to the neural network.
     *
     * @return the number of weights in the neural network.
     */
    public int getNumberWeights() {
        return numberWeights;
    }

    /**
     * This resets all the values that are modified in the forward pass and 
     * backward pass and need to be reset to 0 before doing another
     * forward and backward pass (i.e., all the non-weights/biases).
     */
    public void reset() {
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                layers[layer][number].reset();
            }
        }
    }

    /**
     * This resets the running averages for batch normalization
     * across all the nodes at the beginning of an epoch.
     */
    public void resetRunning() {
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                layers[layer][number].resetRunning();
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the ConvolutionalNeuralNetwork.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public double[] getWeights() throws NeuralNetworkException {
        double[] weights = new double[numberWeights];

        //What we're going to do here is fill in the weights array
        //we just created by having each node set the weights starting
        //at the position variable we're creating. The Node.getWeights
        //method will set the weights variable passed as a parameter,
        //and then return the number of weights it set. We can then
        //use this to increment position so the next node gets weights
        //and puts them in the right position in the weights array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].getWeights(position, weights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the ConvolutionalNeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking. 
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the ConvolutionalNeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                Log.trace("setting weights for layer: " + layer + ", nodeNumber: " + nodeNumber + ", position: " + position);
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the ConvolutionalNeuralNetwork.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public double[] getDeltas() throws NeuralNetworkException {
        double[] deltas = new double[numberWeights];

        //What we're going to do here is fill in the deltas array
        //we just created by having each node set the deltas starting
        //at the position variable we're creating. The Node.getDeltas
        //method will set the deltas variable passed as a parameter,
        //and then return the number of deltas it set. We can then
        //use this to increment position so the next node gets deltas
        //and puts them in the right position in the deltas array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nDeltas = layers[layer][nodeNumber].getDeltas(position, deltas);
                position += nDeltas;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This initializes the weights in the RNN using either Xavier or
     * Kaiming initialization.
    *
     * @param type will be either "xavier" or "kaiming" and this will
     * initialize the child nodes accordingly, using their helper methods.
     * @param bias is the value to set the bias of each node to.
     */
    public void initializeRandomly(String type, double bias) throws NeuralNetworkException {
        if (type.equals("xavier")) {
            int layer = 1;
            while (layer < layers.length) {
                int node = 0;
                int fanIn = 0;
                int fanOut = 0;
                while (node < layers[layer].length) {
                    fanIn += layers[layer][node].fanIn();
                    fanOut += layers[layer][node].fanOut();
                    node++;
                }
                // Initialize weights
                node = 0;
                while (node < layers[layer].length) {
                    layers[layer][node].initializeWeightsAndBiasXavier(bias, fanIn, fanOut);
                    node++;
                }
                layer++;
            }
        } else if (type.equals("kaiming")) {
            int layer = 1;
            while (layer < layers.length) {
                int node = 0;
                int fanIn = 0;
                while (node < layers[layer].length) {
                    fanIn += layers[layer][node].fanIn();
                    node++;
                }
                // Initialize weights
                node = 0;
                while (node < layers[layer].length) {
                    layers[layer][node].initializeWeightsAndBiasKaiming(bias, fanIn);
                    node++;
                }
                layer++;
            }
       } else {
           throw new NeuralNetworkException("Unknown initialization method: " + type);
       }

   }



    /**
     * This performs a forward pass through the neural network given
     * inputs from the input instance.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     *
     * @return the sum of the output of all output nodes
     */
    public double forwardPass(ImageDataSet imageDataSet, int startIndex, int batchSize, boolean training) throws NeuralNetworkException {
        //be sure to reset before doing a forward pass
        reset();

        //set input values differently for time series and character sequences

        //set the input nodes for each time step in the CharacterSequence

        List<Image> images = imageDataSet.getImages(startIndex, batchSize);

        for (int number = 0; number < layers[0].length; number++) {
            ConvolutionalNode inputNode = layers[0][number];
            inputNode.setValues(images, imageDataSet.getChannelAvgs(), imageDataSet.getChannelStdDevs(imageDataSet.getChannelAvgs()));
        }

        //TODO: You need to implement propagating forward for each node (output nodes need
        //to be propagated forward for their recurrent connections to further time steps)
        //for Programming Assignment 3 - Part 1
        //NOTE: This shouldn't need to be changed for Programming Assignment 2 - Parts 2 or 3
        

        //Propogate images forward
        int layer = 0;
        while (layer < layers.length) {
            int node = 0;
            while (node < layers[layer].length) {
                layers[layer][node].propagateForward(training);
                node++;
            }
            layer++;
        }



        //The following is needed for Programming Assignment 2 - Part 1
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        //note that the target value for any time step is the sequence value at that time step + 1
        //this means you should only go up to length - 1 time steps in calculating the loss
        double lossSum = 0;

        if (lossFunction == LossFunction.SVM) {
            //TODO: Implement this for Programming Assignment 3 - Part 1, be sure
            //to calculate for each image in the batch
            int image = 0;
            while (image < batchSize) {
                double wrongOutputDeltaSum = 0;
                int correctNode = images.get(image).label;
                int outNode = 0;
                while (outNode < nOutputs) {
                    if (outNode != correctNode) {
                        ConvolutionalNode currNode = layers[outputLayer][outNode];
                        double testVal = currNode.outputValues[image][0][0][0] - layers[outputLayer][correctNode].outputValues[image][0][0][0] + 1;
                        if (testVal > 0) {
                            lossSum += testVal;
                        }
                        currNode.delta[image][0][0][0] += (testVal < 0) ? 0 : 1.0;
                        wrongOutputDeltaSum -= currNode.delta[image][0][0][0];
                    }
                    outNode++;
                }
                layers[outputLayer][correctNode].delta[image][0][0][0] += wrongOutputDeltaSum;
                image++;
            }

        } else if (lossFunction == LossFunction.SOFTMAX) {
            int image = 0;
            while (image < batchSize) {
                double softmaxSum = 0;
                double maxZ = layers[outputLayer][0].outputValues[image][0][0][0];
                int correctNode = images.get(image).label;
                int outNode = 1;
                while (outNode < nOutputs) {
                    ConvolutionalNode currNode = layers[outputLayer][outNode];
                    if (currNode.outputValues[image][0][0][0] > maxZ) {
                        maxZ = currNode.outputValues[image][0][0][0];
                    }
                    outNode++;
                }
                outNode = 0;
                while (outNode < nOutputs) {
                    ConvolutionalNode currNode = layers[outputLayer][outNode];
                    softmaxSum += Math.exp(currNode.outputValues[image][0][0][0] - maxZ);
                    outNode++;
                }
                outNode = 0;
                while (outNode < nOutputs) {
                    ConvolutionalNode currNode = layers[outputLayer][outNode];

                    currNode.delta[image][0][0][0] += Math.exp(currNode.outputValues[image][0][0][0] - maxZ) / softmaxSum;
                    if (outNode == correctNode) {
                        lossSum -= Math.log(currNode.delta[image][0][0][0]);
                        currNode.delta[image][0][0][0] -= 1;
                    }
                    outNode++;
                }
                image++;
          }

        } else {
            throw new NeuralNetworkException("Could not do a CharacterSequence forward pass on ConvolutionalNeuralNetwork because lossFunction was unknown or invalid: " + lossFunction);
        }

        return lossSum;
    }

    /**
     * This does forward passes over the entire image data set to calculate
     * the total error and accuracy (this is used by GradientDescent.java). We
     * do them both here to improve performance.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param accuracyAndError is a double array of length 2, index 0 will
     * be the accuracy and index 1 will be the error
     */
    public void calculateAccuracyAndError(ImageDataSet imageDataSet, int batchSize, double[] accuracyAndError) throws NeuralNetworkException {
        //TODO: need to implement this for Programming Assignment 3 - Part 2 - Calculate Accuracy and Error
        //the output node with the maximum value is the predicted class
        //you need to sum up how many of these match the actual class
        //for each time step of each sequence, and then calculate: 
        //num correct / total
        //to get a percentage accuracy
        accuracyAndError[0] = 0;
        accuracyAndError[1] = 0;
        ConvolutionalNode[] outputLayer = layers[layers.length - 1];
        int startIndex = 0;
        while (startIndex < imageDataSet.getNumberImages()) {
            accuracyAndError[1] += forwardPass(imageDataSet, startIndex, batchSize, false);
            int image = 0;
            while (image < batchSize) {
                int correctNode = imageDataSet.getImages(startIndex + image, 1).get(0).label;
                int predictedNode = 0;
                double maxValue = outputLayer[0].outputValues[image][0][0][0];
                int node = 1;
                while (node < outputLayer.length) {
                    double testVal = outputLayer[node].outputValues[image][0][0][0];
                    if (testVal > maxValue) {
                        maxValue = testVal;
                        predictedNode = node;
                    }
                    node++;
                }
                accuracyAndError[0] += (predictedNode == correctNode) ? 1 : 0;
                image++;
            }
            startIndex += batchSize;
        }
        accuracyAndError[0] = accuracyAndError[0] / imageDataSet.getNumberImages();
    }


    /**
     * This gets the output values of the neural network 
     * after a forward pass, this will be a 1 dimensional array, one
     * value for each output node
     *
     * @param batchSize is the batch size of for this CNN
     *
     * @return a one dimensional array of the output values from this neural network for
     */
    public double[][] getOutputValues(int batchSize) {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[][] outputValues = new double[batchSize][nOutputs];

        for (int i = 0; i < batchSize; i++) {
            for (int number = 0; number < nOutputs; number++) {
                outputValues[i][number] = layers[outputLayer][number].outputValues[i][0][0][0];
            }
        }

        return outputValues;
    }

    /**
     * The step size used to calculate the gradient numerically using the finite
     * difference method.
     */
    private static final double H = 0.0000001;

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     */
    public double[] getNumericGradient(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 3 - Part 2
        double[] weights = getWeights();
        double[] numericGrad = new double[weights.length];
        int weight = 0;
        while (weight < weights.length) {
            weights[weight] += H;
            setWeights(weights);
            double fPlus = forwardPass(imageDataSet, startIndex, batchSize, true);
            weights[weight] = weights[weight] - H - H;
            setWeights(weights);
            double fMinus = forwardPass(imageDataSet, startIndex, batchSize, true);
            numericGrad[weight] = (fPlus - fMinus) / (2.0 * H);
            weights[weight] += H;
            weight++;
        }
        setWeights(weights);
        return numericGrad;

        // throw new NeuralNetworkException("ConvolutionalNeuralNetwork.getNumericGradient not implemented!");
    }


    /**
     * This performs a backward pass through the neural network given 
     * outputs from the given instance. This will set the deltas in
     * all the edges and nodes which will be used to calculate the 
     * gradient and perform backpropagation.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     *
     */
    public void backwardPass(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 3 - Part 2
        int layer = layers.length - 1;
        while (layer >= 0) {
            int node = layers[layer].length - 1;
            while (node >= 0) {
                layers[layer][node].propagateBackward();
                node--;
            }
            layer--;
        }
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the ConvolutionalNeuralNetwork.backwardPass(Sequence)) Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     */
    public double[] getGradient(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {
        forwardPass(imageDataSet, startIndex, batchSize, true /*we're training here so use the training versions of batch norm and dropout*/);
        backwardPass(imageDataSet, startIndex, batchSize);

        return getDeltas();
    }

    /**
     * Print out numeric vs backprop gradients in a clean manner so that
     * you can see where gradients were not the same
     *
     * @param numericGradient is a previously calculated numeric gradient
     * @param backpropGradient is a previously calculated gradient from backprop
     */
    public void printGradients(double[] numericGradient, double[] backpropGradient) {
        int current = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                current += layers[layer][number].printGradients(current, numericGradient, backpropGradient);
            }
        }
    }
}
