/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 2 - Part 3.
 *
 */
import java.util.Arrays;
import java.util.List;

import data.DataSet;
import data.SequenceDataSet;
import data.TimeSeriesDataSet;
import data.Sequence;
import data.CharacterSequence;
import data.TimeSeries;

import network.LossFunction;
import network.RecurrentNeuralNetwork;
import network.NeuralNetworkException;
import network.RNNNodeType;

import util.Log;
import util.Vector;


public class GradientDescent {
    public static void helpMessage() {
        Log.info("Usage:");
        Log.info("\tjava GradientDescent <data set> <rnn node type> <network type> <initialization type> <gradient descent type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <low threshold> <high threshold> <layer_size_1 ... layer_size_n>");
        Log.info("\t\tdata set can be: 'penn_small', 'penn_full' or 'flights_small', 'flights_full'");
        Log.info("\t\trnn node type can be: 'linear', 'sigmoid', 'tanh', 'lstm', 'gru', 'ugrnn', 'mgu' or 'delta'");
        Log.info("\t\tnetwork type can be: 'feed_forward', 'jordan' or 'elman'");
        Log.info("\t\tinitialization type can be: 'xavier' or 'kaiming'");
        Log.info("\t\tgradient descent type can be: 'stochastic', 'minibatch' or 'batch'");
        Log.info("\t\tbatch size should be > 0. Will be ignored for stochastic or batch gradient descent");
        Log.info("\t\tloss function can be: 'l1_norm', 'l2_norm', 'svm' or 'softmax'");
        Log.info("\t\tepochs is an integer > 0");
        Log.info("\t\tbias is a double");
        Log.info("\t\tlearning rate is a double usually small and > 0");
        Log.info("\t\tmu is a double < 1 and typical values are 0.5, 0.9, 0.95 and 0.99");
        Log.info("\t\tlow threshold is a double value to use as the threshold for gradient boosting (0.05 recommended), if it is < 0, gradient boosting will not be used");
        Log.info("\t\thigh threshold is a double value to use as the threshold for gradient scaling (1.0 recommended), if it is < 0, gradient scaling will not be used");
        Log.info("\t\tlayer_size_1..n is a list of integers which are the number of nodes in each hidden layer");
    }
    public static void scaleGradient(double[] grad, double l, double h){
        double l2 = 0;
        for(int i = 0; i < grad.length; i++){
            l2 += grad[i]*grad[i];
        }
        l2 = Math.sqrt(l2);
        if(l2 < l){
            for(int i = 0; i < grad.length; i++){
                grad[i] = grad[i]*l/l2;
            }
        }
        else if(l2 > h){
            for(int i = 0; i < grad.length; i++){
                grad[i] = grad[i]*h/l2;
            }
        }
    }
    public static void main(String[] arguments) {
        if (arguments.length < 14) {
            helpMessage();
            System.exit(1);
        }

        String dataSetName = arguments[0];
        String rnnNodeTypeStr = arguments[1];
        String networkType = arguments[2];
        String initializationType = arguments[3];
        String descentType = arguments[4];
        int batchSize = Integer.parseInt(arguments[5]);
        String lossFunctionName = arguments[6];
        int epochs = Integer.parseInt(arguments[7]);
        double bias = Double.parseDouble(arguments[8]);
        double learningRate = Double.parseDouble(arguments[9]);
        double mu = Double.parseDouble(arguments[10]);
        double lowThreshold = Double.parseDouble(arguments[11]);
        double highThreshold = Double.parseDouble(arguments[12]);

        int[] layerSizes = new int[arguments.length - 13]; // the remaining arguments are the layer sizes
        for (int i = 13; i < arguments.length; i++) {
            layerSizes[i - 13] = Integer.parseInt(arguments[i]);
        }

        //the and, or and xor datasets will have 1 output (the number of output columns)
        //but the iris and mushroom datasets will have the number of output classes
        int outputLayerSize = 0;

        DataSet trainingDataSet = null;
        DataSet testingDataSet = null;

        if (dataSetName.equals("penn_small")) {
            trainingDataSet = new SequenceDataSet("sequence test set", "./datasets/penntreebank_train_small.txt");
            testingDataSet = new SequenceDataSet("sequence test set", "./datasets/penntreebank_test_small.txt");

        } else if (dataSetName.equals("penn_full")) {
            trainingDataSet = new SequenceDataSet("sequence test set", "./datasets/penntreebank_train_full.txt");
            testingDataSet = new SequenceDataSet("sequence test set", "./datasets/penntreebank_test_full.txt");

        } else if (dataSetName.equals("flights_small")) {
            //TODO: Programming Assignment 2 - Part 3: Make sure you implement the getMins, getMaxes,
            // and normalizeMinMax methods in TimeSeriesDataSet to get this to work.
            trainingDataSet = new TimeSeriesDataSet("flights data training small",
                    new String[]{"./datasets/flight_0_short.csv", "./datasets/flight_1_short.csv", "./datasets/flight_2_short.csv"}, /* input file names */
                    new String[]{"AltAGL", "E1 RPM", "IAS", "LatAc", "NormAc", "Pitch", "Roll"}, /*parameter names for RNN input values */
                    new String[]{"Pitch", "Roll"} /*parameter names for RNN target output values */
            );

            testingDataSet = new TimeSeriesDataSet("flights data testing small",
                    new String[]{"./datasets/flight_3_short.csv"}, /* input file names */
                    new String[]{"AltAGL", "E1 RPM", "IAS", "LatAc", "NormAc", "Pitch", "Roll"}, /*parameter names for RNN input values */
                    new String[]{"Pitch", "Roll"} /*parameter names for RNN target output values */
            );


            double[] mins = ((TimeSeriesDataSet)trainingDataSet).getMins();
            double[] maxs = ((TimeSeriesDataSet)trainingDataSet).getMaxs();

            Log.info("Data set had the following column mins: " + Arrays.toString(mins));
            Log.info("Data set had the following column maxs: " + Arrays.toString(maxs));

            ((TimeSeriesDataSet)trainingDataSet).normalizeMinMax(mins, maxs);
            ((TimeSeriesDataSet)testingDataSet).normalizeMinMax(mins, maxs);
            Log.info("normalized the data");

        } else if (dataSetName.equals("flights_full")) {
            //TODO: Programming Assignment 2 - Part 3: Make sure you implement the getMins, getMaxes,
            // and normalizeMinMax methods in TimeSeriesDataSet to get this to work.
            trainingDataSet = new TimeSeriesDataSet("flights data training full",
                    new String[]{"./datasets/flight_0.csv", "./datasets/flight_1.csv", "./datasets/flight_2.csv"}, /* input file names */
                    new String[]{"AltAGL", "E1 RPM", "IAS", "LatAc", "NormAc", "Pitch", "Roll"}, /*parameter names for RNN input values */
                    new String[]{"Pitch", "Roll"} /*parameter names for RNN target output values */
            );

            testingDataSet = new TimeSeriesDataSet("flights data testing full",
                    new String[]{"./datasets/flight_3.csv"}, /* input file names */
                    new String[]{"AltAGL", "E1 RPM", "IAS", "LatAc", "NormAc", "Pitch", "Roll"}, /*parameter names for RNN input values */
                    new String[]{"Pitch", "Roll"} /*parameter names for RNN target output values */
            );


            double[] mins = ((TimeSeriesDataSet)trainingDataSet).getMins();
            double[] maxs = ((TimeSeriesDataSet)trainingDataSet).getMaxs();

            Log.info("Data set had the following column mins: " + Arrays.toString(mins));
            Log.info("Data set had the following column maxs: " + Arrays.toString(maxs));

            ((TimeSeriesDataSet)trainingDataSet).normalizeMinMax(mins, maxs);
            ((TimeSeriesDataSet)testingDataSet).normalizeMinMax(mins, maxs);
            Log.info("normalized the data");

        } else {
            Log.fatal("unknown data set : " + dataSetName);
            System.exit(1);
        }

        RNNNodeType rnnNodeType = RNNNodeType.LINEAR;
        if (rnnNodeTypeStr.equals("linear")) {
            Log.info("Using an LINEAR RNN node type.");
            rnnNodeType = RNNNodeType.LINEAR;
        } else if (rnnNodeTypeStr.equals("sigmoid")) {
            Log.info("Using an SIGMOID RNN node type.");
            rnnNodeType = RNNNodeType.SIGMOID;
        } else if (rnnNodeTypeStr.equals("tanh")) {
            Log.info("Using an TANHRNN node type.");
            rnnNodeType = RNNNodeType.TANH;
        } else if (rnnNodeTypeStr.equals("LSTM")) {
            Log.info("Using an LSTM RNN node type.");
            rnnNodeType = RNNNodeType.LSTM;
        } else if (rnnNodeTypeStr.equals("GRU")) {
            Log.info("Using an GRU RNN node type.");
            rnnNodeType = RNNNodeType.GRU;
        } else if (rnnNodeTypeStr.equals("UGRNN")) {
            Log.info("Using an UGRNN RNN node type.");
            rnnNodeType = RNNNodeType.UGRNN;
        } else if (rnnNodeTypeStr.equals("MGU")) {
            Log.info("Using an MGU RNN node type.");
            rnnNodeType = RNNNodeType.MGU;
        } else if (rnnNodeTypeStr.equals("DELTA")) {
            Log.info("Using an DELTA RNN node type.");
            rnnNodeType = RNNNodeType.DELTA;
        } else {
            Log.fatal("unknown RNN node type: " + rnnNodeTypeStr);
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

        RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(trainingDataSet.getNumberInputs(), layerSizes, trainingDataSet.getNumberOutputs(), Math.max(trainingDataSet.getMaxLength(), testingDataSet.getMaxLength()), rnnNodeType, lossFunction);
        try {
            rnn.connectFully();

            if (networkType.equals("jordan")) {
                rnn.connectJordan(1 /*use a timeSkip of 1*/);
            } else if (networkType.equals("elman")) {
                rnn.connectElman(1 /*use a timeSkip of 1*/);
            }
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

            rnn.initializeRandomly(initializationType, bias);

            //TODO: For Programming Assignment 2 - Part 3 use this and implement nesterov momentum
            //java will initialize each element in the array to 0
            double[] velocity = new double[rnn.getNumberWeights()];
            double[] velocityPrev = new double[velocity.length];

            double bestError = 10000;
            double error = rnn.forwardPass(trainingDataSet.getSequences()) / trainingDataSet.getNumberSequences();
            double testingError = rnn.forwardPass(testingDataSet.getSequences()) / testingDataSet.getNumberSequences();

            if (trainingDataSet instanceof SequenceDataSet) {
                double accuracy = rnn.calculateAccuracy(((SequenceDataSet)trainingDataSet).getCharacterSequences());
                double testingAccuracy = rnn.calculateAccuracy(((SequenceDataSet)testingDataSet).getCharacterSequences());

                if (error < bestError) bestError = error;
                System.out.println("  " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make the accuracy a percentage*/ + " " + testingError + String.format("%10.5f", testingAccuracy * 100.0) /* make the test accuracy a percentage */);
            } else {
                if (error < bestError) bestError = error;
                System.out.println("  " + bestError + " " + error + " " + testingError);
            }

            for (int i = 0; i < epochs; i++) {

                if (descentType.equals("stochastic")) {
                    trainingDataSet.shuffle();
                    int sequenceIndex = 0;
                    while (sequenceIndex < trainingDataSet.getNumberSequences()) {
                        Sequence sequence = trainingDataSet.getSequence(sequenceIndex);
                        double[] gradient = rnn.getGradient(sequence);

                        int z = 0;
                        while (z < gradient.length) {
                            velocityPrev[z] = velocity[z];
                            velocity[z] = mu * velocity[z] - learningRate * gradient[z];
                            gradient[z] = (1 + mu) * velocity[z] - mu * velocityPrev[z];
                            z++;
                        }

                        // Directly implement the gradient scaling within this loop
                        double norm = Math.sqrt(Arrays.stream(gradient).map(x -> x * x).sum());
                        if (norm < lowThreshold) {
                            z = 0;
                            while (z < gradient.length) {
                                gradient[z] = gradient[z] * lowThreshold / norm;
                                z++;
                            }
                        } else if (norm > highThreshold) {
                            z = 0;
                            while (z < gradient.length) {
                                gradient[z] = gradient[z] * highThreshold / norm;
                                z++;
                            }
                        }
                        rnn.setWeights(gradient);
                        sequenceIndex++;
                    }
                } else if (descentType.equals("minibatch")) {
                    trainingDataSet.shuffle();
                    int startIndex = 0;
                    while (startIndex < trainingDataSet.getNumberSequences()) {
                        int endIndex = Math.min(startIndex + batchSize, trainingDataSet.getNumberSequences());
                        List<Sequence> batch = trainingDataSet.getSequences(startIndex, endIndex);
                        double[] gradient = rnn.getGradient(batch);

                        int z = 0;
                        while (z < gradient.length) {
                            velocityPrev[z] = velocity[z];
                            velocity[z] = mu * velocity[z] - learningRate * gradient[z];
                            gradient[z] = (1 + mu) * velocity[z] - mu * velocityPrev[z];
                            z++;
                        }

                        // Gradient scaling logic integrated here
                        double norm = Math.sqrt(Arrays.stream(gradient).map(x -> x * x).sum());
                        if (norm < lowThreshold) {
                            z = 0;
                            while (z < gradient.length) {
                                gradient[z] = gradient[z] * lowThreshold / norm;
                                z++;
                            }
                        } else if (norm > highThreshold) {
                            z = 0;
                            while (z < gradient.length) {
                                gradient[z] = gradient[z] * highThreshold / norm;
                                z++;
                            }
                        }

                        rnn.setWeights(gradient);
                        startIndex += batchSize;
                    }
                } else if (descentType.equals("batch")) {
                    List<Sequence> sequences = trainingDataSet.getSequences();
                    double[] gradient = rnn.getGradient(sequences);
                    int y = 0;
                    while (y < gradient.length) {
                        velocityPrev[y] = velocity[y];
                        velocity[y] = mu * velocity[y] - learningRate * gradient[y];
                        gradient[y] = (1 + mu) * velocity[y] - mu * velocityPrev[y];
                        y++;
                    }

                    // Gradient scaling logic integrated here
                    double norm = Math.sqrt(Arrays.stream(gradient).map(x -> x * x).sum());
                    if (norm < lowThreshold) {
                        y = 0;
                        while (y < gradient.length) {
                            gradient[y] = gradient[y] * lowThreshold / norm;
                            y++;
                        }
                    } else if (norm > highThreshold) {
                        y = 0;
                        while (y < gradient.length) {
                            gradient[y] = gradient[y] * highThreshold / norm;
                            y++;
                        }
                    }

                    rnn.setWeights(gradient);


                } else {
                    Log.fatal("unknown descent type: " + descentType);
                    helpMessage();
                    System.exit(1);
                }

                //Log.info("weights: " + Arrays.toString(nn.getWeights()));

                //at the end of each epoch, calculate the error over the entire
                //set of sequences and print it out so we can see if we're decreasing
                //the overall error, also do this for the test data to see how we're
                //doing on unseen data
                error = rnn.forwardPass(trainingDataSet.getSequences()) / trainingDataSet.getNumberSequences();
                testingError = rnn.forwardPass(testingDataSet.getSequences()) / testingDataSet.getNumberSequences();

                if (trainingDataSet instanceof SequenceDataSet) {
                    double accuracy = rnn.calculateAccuracy(((SequenceDataSet)trainingDataSet).getCharacterSequences());
                    double testingAccuracy = rnn.calculateAccuracy(((SequenceDataSet)testingDataSet).getCharacterSequences());

                    if (error < bestError) bestError = error;
                    System.out.println("  " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make the accuracy a percentage*/ + " " + testingError + String.format("%10.5f", testingAccuracy * 100.0) /* make the test accuracy a percentage */);
                } else {
                    if (error < bestError) bestError = error;
                    System.out.println("  " + bestError + " " + error + " " + testingError);
                }

            }

        } catch (NeuralNetworkException e) {
            Log.fatal("gradient descent failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}

