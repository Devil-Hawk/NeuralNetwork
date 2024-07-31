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

import static java.lang.Math.tanh;

public class LSTMNode extends RecurrentNode {

    //these are the weight values for the LSTM node
    double wi;
    double wf;
    double wc;
    double wo;

    double ui;
    double uf;
    double uo;

    //these are the bias values for the LSTM node
    double bf;
    double bi;
    double bc;
    double bo;

    //these are the deltas for the weights and biases
    public double delta_wi;
    public double delta_wf;
    public double delta_wc;
    public double delta_wo;

    public double delta_ui;
    public double delta_uf;
    public double delta_uo;

    public double delta_bf;
    public double delta_bi;
    public double delta_bc;
    public double delta_bo;

    //this is the delta value for ct in the diagram, it will be
    //set to the sum of the delta coming in from the outputs (delta)
    //times ot for the time step, plus whatever deltas came in from
    //the subsequent time step during backprop
    public double[] delta_ct;

    //input gate values for each time step
    public double[] it;

    //forward gate values for each time step
    public double[] ft;

    //cell values for each time step
    public double[] ct;

    //output gate values for each time step
    public double[] ot;

    //variable C saved for doing the backward pass
    public double[] C;


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
    public LSTMNode(int layer, int number, NodeType nodeType, int maxSequenceLength) {
        super(layer, number, nodeType, maxSequenceLength, null);

        delta_ct = new double[maxSequenceLength];

        ft = new double[maxSequenceLength];
        it = new double[maxSequenceLength];
        ot = new double[maxSequenceLength];
        ct = new double[maxSequenceLength];
        C = new double[maxSequenceLength];
    }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        //use RecurrentNode's reset to reset everything this has inherited from
        //RecurrentNode, then reset the LSTMNode's fields
        super.reset();
        Log.trace("Resetting LSTM node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            ft[timeStep] = 0;
            it[timeStep] = 0;
            ot[timeStep] = 0;
            ct[timeStep] = 0;
            C[timeStep] = 0;
            delta_ct[timeStep] = 0;
        }

        delta_wi = 0;
        delta_wf = 0;
        delta_wo = 0;
        delta_wc = 0;

        delta_ui = 0;
        delta_uf = 0;
        delta_uo = 0;

        delta_bi = 0;
        delta_bf = 0;
        delta_bo = 0;
        delta_bc = 0;
    }


    /**
     * We need to override the getWeightNames from RecurrentNode as
     * an LSTMNode will have 11 weight and bias names as opposed to
     * just one bias.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weightNames is the array of weight nameswe're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeightNames(int position, String[] weightNames) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            weightNames[position] = "LSTM Node [layer " + layer + ", number " + number + ", wi]";
            weightNames[position + 1] = "LSTM Node [Layer " + layer + ", number " + number + ", wf]";
            weightNames[position + 2] = "LSTM Node [Layer " + layer + ", number " + number + ", wc]";
            weightNames[position + 3] = "LSTM Node [Layer " + layer + ", number " + number + ", wo]";

            weightNames[position + 4] = "LSTM Node [Layer " + layer + ", number " + number + ", ui]";
            weightNames[position + 5] = "LSTM Node [Layer " + layer + ", number " + number + ", uf]";
            weightNames[position + 6] = "LSTM Node [Layer " + layer + ", number " + number + ", uo]";

            weightNames[position + 7] = "LSTM Node [Layer " + layer + ", number " + number + ", bi]";
            weightNames[position + 8] = "LSTM Node [Layer " + layer + ", number " + number + ", bf]";
            weightNames[position + 9] = "LSTM Node [Layer " + layer + ", number " + number + ", bc]";
            weightNames[position + 10] = "LSTM Node [Layer " + layer + ", number " + number + ", bo]";

            weightCount += 11;
        }

        for (Edge edge : outputEdges) {
            String targetType = "";
            if (edge.outputNode instanceof LSTMNode) targetType = "LSTM ";
            weightNames[position + weightCount] = "Edge from LSTM Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            String targetType = "";
            if (recurrentEdge.outputNode instanceof LSTMNode) targetType = "LSTM ";

            weightNames[position + weightCount] = "Recurrent Edge from LSTM Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }


        return weightCount;
    }



    /**
     * We need to override the getWeights from RecurrentNode as
     * an LSTMNode will have 11 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            weights[position] = wi;
            weights[position + 1] = wf;
            weights[position + 2] = wc;
            weights[position + 3] = wo;

            weights[position + 4] = ui;
            weights[position + 5] = uf;
            weights[position + 6] = uo;

            weights[position + 7] = bi;
            weights[position + 8] = bf;
            weights[position + 9] = bc;
            weights[position + 10] = bo;

            weightCount += 11;
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
     * We need to override the getDeltas from RecurrentNode as
     * an LSTMNode will have 11 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        //the first delta set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            deltas[position] = delta_wi;
            deltas[position + 1] = delta_wf;
            deltas[position + 2] = delta_wc;
            deltas[position + 3] = delta_wo;

            deltas[position + 4] = delta_ui;
            deltas[position + 5] = delta_uf;
            deltas[position + 6] = delta_uo;

            deltas[position + 7] = delta_bi;
            deltas[position + 8] = delta_bf;
            deltas[position + 9] = delta_bc;
            deltas[position + 10] = delta_bo;

            deltaCount += 11;
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
     * We need to override the getDeltas from RecurrentNode as
     * an LSTMNode will have 11 weights and biases as opposed to
     * just one bias.
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
        if (nodeType != NodeType.INPUT) {
            wi = weights[position];
            wf = weights[position + 1];
            wc = weights[position + 2];
            wo = weights[position + 3];

            ui = weights[position + 4];
            uf = weights[position + 5];
            uo = weights[position + 6];

            bi = weights[position + 7];
            bf = weights[position + 8];
            bc = weights[position + 9];
            bo = weights[position + 10];

            weightCount += 11;
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

    double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }
    /**
     * This propagates the postActivationValue at this LSTM node
     * to all it's output nodes.
     */
    public void propagateForward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //NOTE: recurrent edges need to be propagated forward from this timeStep to
        //their targetNode at timeStep + the recurrentEdge's timeSkip
        double c_prev;
        if (timeStep == 0) {
            c_prev = 0;
        } else {
            c_prev = ct[timeStep - 1];
        }
        ft[timeStep] = sigmoid(wf*preActivationValue[timeStep] + uf*c_prev + bf);
        it[timeStep] = sigmoid(wi*preActivationValue[timeStep] + ui*c_prev + bi);
        ot[timeStep] = sigmoid(wo*preActivationValue[timeStep] + uo*c_prev + bo);
        C[timeStep] = tanh(wc*preActivationValue[timeStep] + bc);
        ct[timeStep] = ft[timeStep]*c_prev + it[timeStep]*C[timeStep];
        postActivationValue[timeStep] = ot[timeStep]*ct[timeStep];
        int edgeIndex = 0;
        while (edgeIndex < outputEdges.size()) {
            Edge e = outputEdges.get(edgeIndex);
            e.outputNode.preActivationValue[timeStep] += postActivationValue[timeStep]*e.weight;
            edgeIndex++;
        }
        int recurrentEdgeIndex = 0;
        while (recurrentEdgeIndex < outputRecurrentEdges.size()) {
            RecurrentEdge re = outputRecurrentEdges.get(recurrentEdgeIndex);
            if (timeStep + re.timeSkip < preActivationValue.length) {
                re.outputNode.preActivationValue[timeStep + re.timeSkip] += postActivationValue[timeStep]*re.weight;
            }
            recurrentEdgeIndex++;
        }
    }

    /**
     * This propagates the delta back from this node
     * to its incoming edges.
     */
    public void propagateBackward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //node that delta[timeStep] is the delta coming in for the output (delta_ht in the slides)
        double deltaCNext = (timeStep == ct.length - 1) ? 0 : delta_ct[timeStep + 1];
        double previousC = (timeStep == 0) ? 0 : ct[timeStep - 1];
        double deltaSum = deltaCNext + delta[timeStep] * ot[timeStep];
        double deltaIT = it[timeStep] * (1.0 - it[timeStep]);
        double deltaFT = ft[timeStep] * (1.0 - ft[timeStep]);
        double deltaOT = ot[timeStep] * (1.0 - ot[timeStep]);
        double deltaC = 1.0 - C[timeStep] * C[timeStep];
        delta_bi += deltaSum * C[timeStep] * deltaIT;
        delta_bf += deltaSum * previousC * deltaFT;
        delta_bo += delta[timeStep] * ct[timeStep] * deltaOT;
        delta_bc += deltaSum * it[timeStep] * deltaC;
        delta_ui += deltaSum * C[timeStep] * deltaIT * previousC;
        delta_uf += deltaSum * previousC * deltaFT * previousC;
        delta_uo += delta[timeStep] * ct[timeStep] * deltaOT * previousC;
        delta_wi += deltaSum * C[timeStep] * deltaIT * preActivationValue[timeStep];
        delta_wf += deltaSum * previousC * deltaFT * preActivationValue[timeStep];
        delta_wo += delta[timeStep] * ct[timeStep] * deltaOT * preActivationValue[timeStep];
        delta_wc += deltaSum * it[timeStep] * deltaC * preActivationValue[timeStep];
        delta_ct[timeStep] = deltaSum * ft[timeStep]
                + deltaSum * C[timeStep] * deltaIT * ui
                + deltaSum * previousC * deltaFT * uf
                + delta[timeStep] * ct[timeStep] * deltaOT * uo;
        delta[timeStep] = deltaSum * it[timeStep] * deltaC * wc
                + deltaSum * C[timeStep] * deltaIT * wi
                + deltaSum * previousC * deltaFT * wf
                + delta[timeStep] * ct[timeStep] * deltaOT * wo;
        int inputEdgeIndex = 0;
        while (inputEdgeIndex < inputEdges.size()) {
            Edge e = inputEdges.get(inputEdgeIndex);
            e.propagateBackward(timeStep, delta[timeStep]);
            inputEdgeIndex++;
        }

        int inputRecurrentEdgeIndex = 0;
        while (inputRecurrentEdgeIndex < inputRecurrentEdges.size()) {
            RecurrentEdge r = inputRecurrentEdges.get(inputRecurrentEdgeIndex);
            r.propagateBackward(timeStep, delta[timeStep]);
            inputRecurrentEdgeIndex++;
        }
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
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        Random weightGen = new Random();
        ui = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        uf = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        uo = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        bf = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian() + 1;
        bi = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        bc = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        bo = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        wi = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        wf = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        wc = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
        wo = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();

        int index = 0;
        while(index < inputEdges.size()) {
            Edge e = inputEdges.get(index);
            e.weight = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
            index++;
        }

        index = 0;
        while(index < inputRecurrentEdges.size()) {
            RecurrentEdge re = inputRecurrentEdges.get(index);
            re.weight = Math.sqrt(2.0/fanIn) * weightGen.nextGaussian();
            index++;
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
        Random weightGen = new Random();

        uf = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        uo = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        ui = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        bf = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1) + 1;
        bi = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        bc = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        bo = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        wi = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        wf = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        wc = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
        wo = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);

        int i = 0;
        while(i < inputEdges.size()) {
            Edge e = inputEdges.get(i);
            e.weight = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
            i++;
        }

        i = 0;
        while(i < inputRecurrentEdges.size()) {
            RecurrentEdge re = inputRecurrentEdges.get(i);
            re.weight = Math.sqrt(6.0 / (fanIn + fanOut)) * (2.0 * weightGen.nextDouble() - 1);
            i++;
        }
    }


    /**
     * Prints concise information about this node.
     *
     * @return The node as a short string.
     */
    public String toString() {
        return "[LSTM Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}
