/**
 * This class represents a Node in the neural network. It will
 * have a list of all input and output edges, as well as its
 * own value. It will also track it's layer in the network and
 * if it is an input, hidden or output node.
 */
package network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import util.Log;

public class RecurrentNode {
    //the layer of this node in the neural network. This is set 
    //final so we cannot change it after assigning it (which 
    //could cause bugs)
    protected final int layer;

    //the number of this node in it's layer. this is mostly
    //just used for printing friendly error message
    protected final int number;

    //this is the maximum sequence length of any input sequence
    //this RNN can train on - this gives us a limit to how long
    //our different arrays should be
    protected final int maxSequenceLength;

    //the type of this node (input, hidden or output). This 
    //is set final so we cannot change it after assigning it 
    //(which could cause bugs)
    protected final NodeType nodeType;

    //the activation function this node will use, can be
    //either sigmoid, tanh or softmax (for the output
    //layer).
    protected final ActivationType activationType;

    //this is the value which is calculated by the forward
    //pass (if it is hidden or output), or assigned by the
    //data set if it is an input node, before the activation
    //function is applied
    public double[] preActivationValue;

    //this is the value which is calculated by the forward
    //pass (if it is not an input node) after the activation
    //function has been applied.
    public double[] postActivationValue;

    //this is the delta/error calculated by backpropagation
    public double[] delta;

    //this is the bias value added to the sum of the inputs
    //multiplied by the weights before the activation function
    //is applied
    protected double bias;

    //thius is the delta/error calculated by backpropagation
    //for the bias
    protected double biasDelta;

    //this is a list of all incoming edges to this node
    protected List<Edge> inputEdges;

    //this is a list of all outgoing edges from this node
    protected List<Edge> outputEdges;

    //this is a list of all incoming edges to this node
    protected List<RecurrentEdge> inputRecurrentEdges;

    //this is a list of all outgoing edges from this node
    protected List<RecurrentEdge> outputRecurrentEdges;



    /**
     * This creates a new node at a given layer in the
     * network and specifies it's type (either input,
     * hidden, our output).
     *
     * @param layer is the layer of the Node in
     * the neural network
     * @param type is the type of node, specified by
     * the Node.NodeType enumeration.
     */
    public RecurrentNode(int layer, int number, NodeType nodeType, int maxSequenceLength, ActivationType activationType) {
        this.layer = layer;
        this.number = number;
        this.nodeType = nodeType;
        this.maxSequenceLength = maxSequenceLength;
        this.activationType = activationType;

        preActivationValue = new double[maxSequenceLength];
        postActivationValue = new double[maxSequenceLength];
        delta = new double[maxSequenceLength];

        //initialize the input and output edges lists
        //as ArrayLists
        inputEdges = new ArrayList<Edge>();
        outputEdges = new ArrayList<Edge>();

        inputRecurrentEdges = new ArrayList<RecurrentEdge>();
        outputRecurrentEdges = new ArrayList<RecurrentEdge>();

        Log.trace("Created a node: " + toString());
    }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        Log.trace("Resetting node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            preActivationValue[timeStep] = 0;
            postActivationValue[timeStep] = 0;
            delta[timeStep] = 0;
        }

        biasDelta = 0;

        for (Edge outputEdge : outputEdges) {
            outputEdge.weightDelta = 0;
        }

        for (RecurrentEdge outputRecurrentEdge : outputRecurrentEdges) {
            outputRecurrentEdge.weightDelta = 0;
        }
    }


    /**
     * The Edge class will call this on its output Node when it is
     * constructed (the input and output nodes of an edge
     * are passed as parameters to the Edge constructor).
     *
     * @param outgoingEdge the new outgoingEdge to add
     *
     * @throws NeuralNetworkException if the edge already exists
     */
    public void addOutgoingEdge(Edge outgoingEdge) throws NeuralNetworkException {
        //lets have a sanity check to make sure we don't duplicate adding
        //an edge
        for (Edge edge : outputEdges) {
            if (edge.equals(outgoingEdge)) {
                throw new NeuralNetworkException("Attempted to add an outgoing edge to node " + toString()
                        + " but could not as it already had an edge to the same output node: " + edge.outputNode.toString());
            }
        }

        Log.trace("Node " + toString() + " added outgoing edge to Node " + outgoingEdge.outputNode);
        outputEdges.add(outgoingEdge);
    }

    /**
     * The Edge class will call this on its input Node when it is
     * constructed (the input and output nodes of an edge
     * are passed as parameters to the Edge constructor).
     *
     * @param incomingEdge the new incomingEdge to add
     *
     * @throws NeuralNetworkException if the edge already exists
     */
    public void addIncomingEdge(Edge incomingEdge) throws NeuralNetworkException {
        //lets have a sanity check to make sure we don't duplicate adding
        //an edge
        for (Edge edge : inputEdges) {
            if (edge.equals(incomingEdge)) {
                throw new NeuralNetworkException("Attempted to add an incoming edge to node " + toString()
                        + " but could not as it already had an edge to the same input node: " + edge.inputNode.toString());
            }
        }

        Log.trace("Node " + toString() + " added incoming edge from Node " + incomingEdge.inputNode);
        inputEdges.add(incomingEdge);
    }

    /**
     * The RecurrentEdge class will call this on its output Node when it is
     * constructed (the input and output nodes of an edge
     * are passed as parameters to the RecurrentEdge constructor).
     *
     * @param outgoingRecurrentEdge the new outgoingRecurrentEdge to add
     *
     * @throws NeuralNetworkException if the edge already exists
     */
    public void addOutgoingRecurrentEdge(RecurrentEdge outgoingRecurrentEdge) throws NeuralNetworkException {
        //lets have a sanity check to make sure we don't duplicate adding
        //an edge
        for (RecurrentEdge edge : outputRecurrentEdges) {
            if (edge.equals(outgoingRecurrentEdge)) {
                throw new NeuralNetworkException("Attempted to add an outgoing edge to node " + toString()
                        + " but could not as it already had an edge to the same output node: " + edge.outputNode.toString());
            }
        }

        Log.trace("Node " + toString() + " added outgoing edge to Node " + outgoingRecurrentEdge.outputNode);
        outputRecurrentEdges.add(outgoingRecurrentEdge);
    }

    /**
     * The RecurrentEdge class will call this on its input Node when it is
     * constructed (the input and output nodes of an edge
     * are passed as parameters to the RecurrentEdge constructor).
     *
     * @param incomingRecurrentEdge the new incomingRecurrentEdge to add
     *
     * @throws NeuralNetworkException if the edge already exists
     */
    public void addIncomingRecurrentEdge(RecurrentEdge incomingRecurrentEdge) throws NeuralNetworkException {
        //lets have a sanity check to make sure we don't duplicate adding
        //an edge
        for (RecurrentEdge edge : inputRecurrentEdges) {
            if (edge.equals(incomingRecurrentEdge)) {
                throw new NeuralNetworkException("Attempted to add an incoming edge to node " + toString()
                        + " but could not as it already had an edge to the same input node: " + edge.inputNode.toString());
            }
        }

        Log.trace("Node " + toString() + " added incoming edge from Node " + incomingRecurrentEdge.inputNode);
        inputRecurrentEdges.add(incomingRecurrentEdge);
    }


    /**
     * Used to get the name of the weights of this node along with the name of the weights
     * of all of it's outgoing edges. It will set the weight names in the weightNames
     * parameter passed in starting at position, and return the number of
     * weight names it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weightNames is the array of weight names we're setting.
     *
     * @return the number of weights set in the weight names parameter
     */
    public int getWeightNames(int position, String[] weightNames) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            weightNames[position] = "Node [layer " + layer + ", number " + number + "]";
            weightCount = 1;
        }

        for (Edge edge : outputEdges) {
            weightNames[position + weightCount] = "Edge from Node [layer " + layer + ", number " + number + "] to Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            weightNames[position + weightCount] = "Recurrent Edge from Node [layer " + layer + ", number " + number + "] to Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }

        return weightCount;
    }



    /**
     * Used to get the weights of this node along with the weights
     * of all of it's outgoing edges. It will set the weights in the weights
     * parameter passed in starting at position, and return the number of
     * weights it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            weights[position] = bias;
            weightCount = 1;
        }

        for (Edge edge : outputEdges) {
            weights[position + weightCount] = edge.weight;
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            weights[position + weightCount] = recurrentEdge.weight;
            weightCount++;
        }

        return weightCount;
    }

    /**
     * Used to get the deltas of this node along with the deltas
     * of all of it's outgoing edges. It will set the deltas in the deltas
     * parameter passed in starting at position, and return the number of
     * deltas it set.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        //the first delta set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            deltas[position] = biasDelta;
            deltaCount = 1;
        }

        for (Edge edge : outputEdges) {
            deltas[position + deltaCount] = edge.weightDelta;
            deltaCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            deltas[position + deltaCount] = recurrentEdge.weightDelta;
            deltaCount++;
        }

        return deltaCount;
    }


    /**
     * Used to set the weights of this node along with the weights of
     * all it's outgoing edges. It uses the same technique as Node.getWeights
     * where the starting position of weights to set is passed, and it returns
     * how many weights were set.
     * 
     * @param position is the starting position in the weights parameter to start
     * setting weights from.
     * @param weights is the array of weights we are setting from
     *
     * @return the number of weights gotten from the weights parameter
     */

    public int setWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            bias = weights[position];
            weightCount = 1;
        }

        for (Edge edge : outputEdges) {
            edge.weight = weights[position + weightCount];
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            recurrentEdge.weight = weights[position + weightCount];
            weightCount++;
        }

        return weightCount;
    }

    /**
     * This applys the linear activation function to this node at the given time step. The postActivationValue
     * will be set to the preActivationValue.
     *
     * @param timeStep is the timeStep to apply the activation function on
     */
    public void applyLinear(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 1
        this.postActivationValue[timeStep] += this.preActivationValue[timeStep];
    }

    /**
     * This applys the sigmoid function to this node at the given time step. The postActivationValue
     * will be set to sigmoid(preActivationValue).
     *
     * @param timeStep is the timeStep to apply the activation function on
     */
    public void applySigmoid(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 1
        this.postActivationValue[timeStep] += 1.0/(1.0+Math.exp(-preActivationValue[timeStep]));
    }

    /**
     * This applys the tanh function to this node at the given time step. The postActivationValue
     * will be set to tanh(preActivationValue).
     *
     * @param timeStep is the timeStep to apply the activation function on
     */
    public void applyTanh(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 1
        this.postActivationValue[timeStep] += Math.tanh(this.preActivationValue[timeStep]);
    }

    /**
     * This propagates the postActivationValue at this node
     * to all it's output nodes.
     */
    public void propagateForward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 1
        //NOTE: recurrent edges need to be propagated forward from this timeStep to
        //their targetNode at timeStep + the recurrentEdge's timeSkip
        if (this.nodeType == NodeType.HIDDEN) {
            this.preActivationValue[timeStep] += this.bias;
        }

        switch (this.activationType) {
            case LINEAR:
                this.applyLinear(timeStep);
                break;
            case SIGMOID:
                this.applySigmoid(timeStep);
                break;
            case TANH:
                this.applyTanh(timeStep);
                break;
            default:
                // Handle default case
                break;
        }

        int edgeIndex = 0;
        while (edgeIndex < outputEdges.size()) {
            Edge edge = outputEdges.get(edgeIndex);
            edge.outputNode.preActivationValue[timeStep] += (edge.weight * this.postActivationValue[timeStep]);
            edgeIndex++;
        }

        int recurrentEdgeIndex = 0;
        while (recurrentEdgeIndex < outputRecurrentEdges.size()) {
            RecurrentEdge recurrentEdge = outputRecurrentEdges.get(recurrentEdgeIndex);
            recurrentEdge.outputNode.preActivationValue[timeStep + recurrentEdge.timeSkip] +=
                    (recurrentEdge.weight * this.postActivationValue[timeStep]);
            recurrentEdgeIndex++;
        }
    }

    /**
     * This propagates the delta back from this node
     * to its incoming edges.
     */
    public void propagateBackward(int timeStep) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        //be sure to sum the biasDelta over all time steps
        //back propagate this nodes delta at this time step into the each recurrent
        //edge as well
        double derivative = 0;
        switch (this.activationType) {
            case LINEAR:
                derivative = 1.0;
                break;
            case SIGMOID:
                derivative = this.postActivationValue[timeStep] * (1 - this.postActivationValue[timeStep]);
                break;
            case TANH:
                derivative = 1 - (this.postActivationValue[timeStep] * this.postActivationValue[timeStep]);
                break;
        }
        double delta = this.delta[timeStep];
        double valuePropagate = delta * derivative;
        int edgeIndex = 0;
        while (edgeIndex < this.inputEdges.size()) {
            Edge incomingEdge = this.inputEdges.get(edgeIndex);
            incomingEdge.propagateBackward(timeStep, valuePropagate);
            edgeIndex++;
        }
        int recurrentEdgeIndex = 0;
        while (recurrentEdgeIndex < this.inputRecurrentEdges.size()) {
            RecurrentEdge incomingRecurrentEdge = this.inputRecurrentEdges.get(recurrentEdgeIndex);
            incomingRecurrentEdge.propagateBackward(timeStep, valuePropagate);
            recurrentEdgeIndex++;
        }
        this.biasDelta += valuePropagate;
    }
    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly initializes each incoming edge weight by using
     *  Random.nextGaussian() / sqrt(N) where N is the number
     *  of incoming edges.
     *
     *  @param bias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasKaiming(int fanIn, double bias) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        int N = fanIn;
        this.bias = bias;
        Random r = new Random();
        int edgeIndex = 0;
        while (edgeIndex < this.inputEdges.size()) {
            Edge e = this.inputEdges.get(edgeIndex);
            double randomWeight = (r.nextGaussian() * Math.sqrt(2)) / Math.sqrt(N);
            e.weight = randomWeight;
            edgeIndex++;
        }
        int recurrentEdgeIndex = 0;
        while (recurrentEdgeIndex < this.inputRecurrentEdges.size()) {
            RecurrentEdge recEdge = this.inputRecurrentEdges.get(recurrentEdgeIndex);
            double randomWeight = (r.nextGaussian() * Math.sqrt(2)) / Math.sqrt(N);
            recEdge.weight = randomWeight;
            recurrentEdgeIndex++;
        }
    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly intializes each incoming edge weight uniformly
     *  at random (you can use Random.nextDouble()) between 
     *  +/- sqrt(6) / sqrt(fan_in + fan_out) 
     *
     *  @param bias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasXavier(int fanIn, int fanOut, double bias) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        double range = Math.sqrt(6) / Math.sqrt(fanIn + fanOut);
        this.bias = bias;
        Random r = new Random();
        double randDouble = (r.nextDouble() * 2 * range) - range;

// Initialize weights for input edges
        int edgeIndex = 0;
        while (edgeIndex < this.inputEdges.size()) {
            Edge e = this.inputEdges.get(edgeIndex);
            double randomWeight = randDouble;
            e.weight = randomWeight;
            edgeIndex++;
        }

// Initialize weights for input recurrent edges
        int recurrentEdgeIndex = 0;
        while (recurrentEdgeIndex < this.inputRecurrentEdges.size()) {
            RecurrentEdge recEdge = this.inputRecurrentEdges.get(recurrentEdgeIndex);
            double randomWeight = randDouble;
            recEdge.weight = randomWeight;
            recurrentEdgeIndex++;
        }
    }


    /**
     * Prints concise information about this node.
     *
     * @return The node as a short string.
     */
    public String toString() {
        return "[Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}
