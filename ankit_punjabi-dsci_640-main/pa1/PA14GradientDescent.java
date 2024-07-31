/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class PA14GradientDescent {
    public static void helpMessage() {
        Log.info("Usage:");
        Log.info("\tjava PA14GradientDescent <data set> <gradient descent type> <batch size> <loss function> <epochs> <bias> <learning rate method> <learning rate> <mu / eps> <b1 / decay rate> <b2> <layer_size_1 ... layer_size_n");
        Log.info("\t\tdata set can be: 'and', 'or' or 'xor', 'iris' or 'mushroom'");
        Log.info("\t\tgradient descent type can be: 'stochastic', 'minibatch' or 'batch'");
        Log.info("\t\tbatch size should be > 0. Will be ignored for stochastic or batch gradient descent");
        Log.info("\t\tloss function can be: 'l1_norm', 'l2_norm', 'svm' or 'softmax'");
        Log.info("\t\tepochs is an integer > 0");
        Log.info("\t\tbias is a double");
        Log.info("\t\tlearning rate method can be: 'nesterov', 'rmsprop' or 'adam'");
        Log.info("\t\tlearning rate is a double usually small and > 0");
        Log.info("\t\tmu is a double < 1 and typical values are 0.5, 0.9, 0.95 and 0.99");
        Log.info("\t\tlayer_size_1..n is a list of integers which are the number of nodes in each hidden layer");
    }

    public static void main(String[] arguments) {
        if (arguments.length < 12) {
            helpMessage();
            System.exit(1);
        }

        String dataSetName = arguments[0];
        String descentType = arguments[1];
        int batchSize = Integer.parseInt(arguments[2]);
        String lossFunctionName = arguments[3];
        int epochs = Integer.parseInt(arguments[4]);
        double bias = Double.parseDouble(arguments[5]);
        String learningRateMethod = arguments[6];
        double learningRate = Double.parseDouble(arguments[7]);
        double mu = Double.parseDouble(arguments[8]);
        double b1 = Double.parseDouble(arguments[9]);
        double b2 = Double.parseDouble(arguments[10]);

        int[] layerSizes = new int[arguments.length - 11]; // the remaining arguments are the layer sizes
        for (int i = 11; i < arguments.length; i++) {
            layerSizes[i - 11] = Integer.parseInt(arguments[i]);
        }

        //the and, or and xor datasets will have 1 output (the number of output columns)
        //but the iris and mushroom datasets will have the number of output classes
        int outputLayerSize = 0;

        DataSet dataSet = null;
        if (dataSetName.equals("and")) {
            dataSet = new DataSet("and data", "./datasets/and.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("or")) {
            dataSet = new DataSet("or data", "./datasets/or.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("xor")) {
            dataSet = new DataSet("xor data", "./datasets/xor.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("iris")) {
            // TODO: PA1-4: Make sure you implement the getInputMeans,
            //getInputStandardDeviations and normalize methods in
            //DataSet to get this to work.
            dataSet = new DataSet("iris data", "./datasets/iris.txt");
            double[] means = dataSet.getInputMeans();
            double[] stdDevs = dataSet.getInputStandardDeviations();
            Log.info("data set means: " + Arrays.toString(means));
            Log.info("data set standard deviations: " + Arrays.toString(stdDevs));
            dataSet.normalize(means, stdDevs);

            outputLayerSize = dataSet.getNumberClasses();
        } else if (dataSetName.equals("mushroom")) {
            dataSet = new DataSet("mushroom data", "./datasets/agaricus-lepiota.txt");
            outputLayerSize = dataSet.getNumberClasses();
        } else {
            Log.fatal("unknown data set : " + dataSetName);
            System.exit(1);
        }

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("l1_norm")) {
            Log.info("Using an L1_NORM loss function.");
            lossFunction = LossFunction.L1_NORM;
        } else if (lossFunctionName.equals("l2_norm")) {
            Log.info("Using an L2_NORM loss function.");
            lossFunction = LossFunction.L2_NORM;
        } else if (lossFunctionName.equals("svm")) {
            Log.info("Using an SVM loss function.");
            lossFunction = LossFunction.SVM;
        } else if (lossFunctionName.equals("softmax")) {
            Log.info("Using an SOFTMAX loss function.");
            lossFunction = LossFunction.SOFTMAX;
        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        NeuralNetwork nn = new NeuralNetwork(dataSet.getNumberInputs(), layerSizes, outputLayerSize, lossFunction);
        try {
            nn.connectFully();
        } catch (NeuralNetworkException e) {
            Log.fatal("ERROR connecting the neural network -- this should not happen!.");
            e.printStackTrace();
            System.exit(1);
        }

        //start the gradient descent
        try {
            Log.info("Starting " + descentType + " gradient descent!");
            if (descentType.equals("minibatch")) {
                Log.info(descentType + "(" + batchSize + "), " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu:" + mu);
            } else {
                Log.info(descentType + ", " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu:" + mu);
            }

            nn.initializeRandomly(bias);

            //TODO: For PA1-4 use this and implement nesterov momentum
            //java will initialize each element in the array to 0
            double[] velocity = new double[nn.getNumberWeights()];
            double[] oldVelocity = new double[velocity.length];

            //TODO: BONUS PA1-4: (1 point) implement the RMSprop
            //per-parameter adaptive learning rate method.
            double[] cache = new double[velocity.length];

            //TODO: BONUS PA1-4: (1 point) implement the Adam
            //per-parameter adaptive learning rate method.
            //For these you will need to add a command line flag
            //to select if which method you'll use (nesterov, rmsprop or adam)
            double[] adamV = new double[velocity.length];
            double[] adamVT = new double[velocity.length];
            double[] adamM = new double[velocity.length];
            double[] adamMT = new double[velocity.length];

            double bestError = 10000;
            double error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
            double accuracy = nn.calculateAccuracy(dataSet.getInstances());

            if (error < bestError) bestError = error;
            System.out.println("  " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make hte accuracy a percentage*/);

            for (int i = 0; i < epochs; i++) {

                if (descentType.equals("stochastic")) {
                    // TODO: PA1-3 you need to implement one epoch (pass through the
                    //training data) for stochastic gradient descent
                    dataSet.shuffle();
                    Iterator<Instance> instanceIterator = dataSet.getInstances().iterator();
                    while (instanceIterator.hasNext()) {
                        Instance instance = instanceIterator.next();
                        double[] weights = nn.getWeights();
                        double[] gradientnew = nn.getGradient(instance);

                        switch (learningRateMethod) {
                            case "nesterov":
                                nesterovUpdate(learningRate, mu, velocity, oldVelocity, weights, gradientnew);
                                break;
                            case "rmsprop":
                                rmspropUpdate(cache, b1, learningRate, mu, weights, gradientnew);
                                break;
                            case "adam":
                                adamUpdate(adamM, b1, gradientnew, adamMT, i, adamV, adamVT, b2, weights, learningRate, mu);
                                break;
                        }

                        nn.setWeights(weights);
                    }
                } else if (descentType.equals("minibatch")) {
                    // TODO: PA1-3 you need to implement one epoch (pass through the
                    //training data) for minibatch gradient descent
                    // shuffle dataset
                    dataSet.shuffle();
                    int instanceIndex = 0;
                    while (instanceIndex < dataSet.getInstances().size()) {
                        double[] gradients = nn.getGradient(dataSet.getInstances(instanceIndex, batchSize));
                        double[] weights = nn.getWeights();

                        switch (learningRateMethod) {
                            case "nesterov":
                                nesterovUpdate(learningRate, mu, velocity, oldVelocity, weights, gradients);
                                break;
                            case "rmsprop":
                                rmspropUpdate(cache, b1, learningRate, mu, weights, gradients);
                                break;
                            case "adam":
                                adamUpdate(adamM, b1, gradients, adamMT, i, adamV, adamVT, b2, weights, learningRate, mu);
                                break;
                        }
                        nn.setWeights(weights);
                        instanceIndex += batchSize;
                    }
                } else if (descentType.equals("batch")) {
                    // TODO: PA1-3 you need to implement one epoch (pass through the training
                    //instances) for batch gradient descent
                    // get gradients for the dataset
                    double[] gradients = nn.getGradient(dataSet.getInstances());
                    double[] weights = nn.getWeights();
                    switch (learningRateMethod) {
                        case "nesterov":
                            nesterovUpdate(learningRate, mu, velocity, oldVelocity, weights, gradients);
                            break;
                        case "rmsprop":
                            rmspropUpdate(cache, b1, learningRate, mu, weights, gradients);
                            break;
                        case "adam":
                            adamUpdate(adamM, b1, gradients, adamMT, i, adamV, adamVT, b2, weights, learningRate, mu);
                            break;
                    }
                    nn.setWeights(weights);
                } else {
                    Log.fatal("unknown descent type: " + descentType);
                    helpMessage();
                    System.exit(1);
                }

                //Log.info("weights: " + Arrays.toString(nn.getWeights()));

                //at the end of each epoch, calculate the error over the entire
                //set of instances and print it out so we can see if we're decreasing
                //the overall error
                error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
                accuracy = nn.calculateAccuracy(dataSet.getInstances());

                if (error < bestError) bestError = error;
                System.out.println(i + " " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make hte accuracy a percentage*/);
            }

        } catch (NeuralNetworkException e) {
            Log.fatal("gradient descent failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void nesterovUpdate(double learningRate, double mu, double[] velocity, double[] velocityOld, double[] weights, double[] gradient){
        for(int i = 0; i < weights.length; i++){
            velocityOld[i] = velocity[i];
            velocity[i] = mu * velocity[i] - learningRate * gradient[i];
            weights[i] += (-1.0 * mu * velocityOld[i]) + ((1 + mu) * velocity[i]);
        }
    }

    private static void rmspropUpdate(double[] cache, double decayRate, double learningRate, double eps, double[] weights, double[] gradient){
        for(int i = 0; i < weights.length; i++){
            cache[i] = decayRate * cache[i] + (1 - decayRate) * Math.pow(gradient[i], 2.0);
            weights[i] -= (learningRate / Math.pow(cache[i] + eps, 0.5)) * gradient[i];
        }
    }

    private static void adamUpdate(double[] m, double b1, double[] gradient, double[] mt, int epochNumber, double[] v, double[] vt, double b2, double[] weights, double learningRate, double eps){
        for(int i = 0; i < weights.length; i++) {
            m[i] = b1 * m[i] + (1 - b1) * gradient[i];
            mt[i] = m[i] / (1 - Math.pow(b1, epochNumber + 1.0));
            v[i] = b2 * v[i] + (1 - b2) * Math.pow(gradient[i], 2.0);
            vt[i] = v[i] / (1 - Math.pow(b2, epochNumber + 1.0));
            weights[i] -= learningRate * (mt[i] / (Math.sqrt(vt[i] + eps)));
        }
    }
}

