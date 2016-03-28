package controllers.machineLearning.neuralNetwok;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.all.Sigmoid;
import controllers.machineLearning.all.TextToArr;
import controllers.machineLearning.linear_logistic.FeatureNormalize;
import controllers.machineLearning.logistic.LogisticPredict;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by shrestha on 12/13/2015.
 */
public class FeedForward {

    private double[][] X;
    private double[][] y;
    private double[][] theta1;
    private double[][] theta2;

    public void initialize(){
        TextToArr textToArr = new TextToArr();
        try {
            File currentDirectory = new File(new File(".").getAbsolutePath());
            String xPath = currentDirectory.getCanonicalPath()+"/src/main/resources/data/feedForward/feedforwardX.txt";
            String yPath = currentDirectory.getCanonicalPath()+"/src/main/resources/data/feedForward/feedforwardY.txt";
//            String theta1Path = currentDirectory.getCanonicalPath()+"/src/main/resources/data/feedForward/theta1.txt";
//            String theta2Path = currentDirectory.getCanonicalPath()+"/src/main/resources/data/feedForward/theta2.txt";
            String strSeperator = "\\s+";
            ArrayList<ArrayList<Double>> xdata = textToArr.convert(xPath, strSeperator);
            ArrayList<ArrayList<Double>> ydata = textToArr.convert(yPath, strSeperator);
//            ArrayList<ArrayList<Double>> theta1data = textToArr.convert(theta1Path, strSeperator);
//            ArrayList<ArrayList<Double>> theta2data = textToArr.convert(theta2Path, strSeperator);
            this.X = textToArr.datatoArr(xdata);
            this.y = textToArr.datatoArr(ydata);
            FeatureNormalize featureNormalize = new FeatureNormalize();
            X = featureNormalize.getFeatureNormalize(X);
//            double[][] X = {{12, 22, 34}, {3, 2, 34}, {1, 2, 3}, {12, 22, 34}, {3, 2, 34}}; //identical to octave y variable
//            double[][] y = {{2}, {3}, {1}, {2}, {3}}; //identical to octave y variable
//            theta1 = textToArr.datatoArr(theta1data);
//            theta2 = textToArr.datatoArr(theta2data);
            /**
             * this theta is optimized theta values given in Ng Andrew not
             */
//            this.theta1 = textToArr.datatoArr(theta1data);
//            this.theta2 = textToArr.datatoArr(theta2data);

            /**** Random Initialize Theta Start ****/
//            RandInitializeWeights randInitializeWeights = new RandInitializeWeights();
//            int theta1Row = 25; //this is the number of output units from hidden layer and it can be any desired number. This initialization means there are theta1Row units in second layer;
//            int theta1Col = this.X[0].length;
//            double epsilon = 0.12; //initialized as per ng Andrew coursera notes
//            double[][] randTheta1 = randInitializeWeights.getRandomWeights(theta1Row, theta1Col, epsilon);
//            this.theta1 = randInitializeWeights.getRandomWeights(theta1Row, theta1Col, epsilon);
//            int theta2Row = 10;
//            int theta2Col = theta1Row;
//            double[][] randTheta2 = randInitializeWeights.getRandomWeights(theta2Row, theta2Col, epsilon);
//            this.theta2 = randInitializeWeights.getRandomWeights(theta2Row, theta2Col, epsilon);
//            System.out.println("theta1 and theta2 are initialized with random values hence it affects the accuracy of the network");
            /**** Random Initialize Theta End ****/

            /***** Cost Start *****/
            int lambda = 0;
            //these inputs for breastcancer data
//            int inputlayersize = 34;
//            int hiddenlayersize = 25;
//            int numlabels = 2;
            //these inputs for number digits
            int inputlayersize = X[0].length;
            int hiddenlayersize = inputlayersize/2;
            int numlabels = 3;
            NNCostFunction nnCostFunction = new NNCostFunction();
            nnCostFunction.setMulticlass(true);
            DebugInitializeWeights debugInitializeWeights = new DebugInitializeWeights();
            double[][] randTheta1 = debugInitializeWeights.createMatrix(hiddenlayersize, inputlayersize);
            double[][] randTheta2 = debugInitializeWeights.createMatrix(numlabels, hiddenlayersize);
            Backwardpropopagation backwardpropopagation = new Backwardpropopagation();
//            double[][] a = {{2, 3, 4}, {5, 3, 4}};
//            double[][] b = {{20, 30}, {50, 30}};
//            double[][] c = backwardpropopagation.combineTheta(a, b);
            double[][] theta = backwardpropopagation.combineTheta(randTheta1, randTheta2);

//            CheckNNGradients checkNNGradients = new CheckNNGradients();
//            checkNNGradients.checkNeuralNetwork(lambda, inputlayersize, hiddenlayersize, numlabels, X, y, randTheta1, randTheta2, theta, true);

            double cost = nnCostFunction.cost(theta,inputlayersize,hiddenlayersize,
            numlabels, X, y, lambda);
            System.out.println("Cost: "+cost);
            /*****Gradient Descent Start*****/
            double alpha = 0.1;
            int iter = 660; //alpha = 0.001 iter = 2000 accuracy: 76%
            GradientDescentBackpropagation gradientDescentBackpropagation = new GradientDescentBackpropagation();
            double[][] gradTheta = gradientDescentBackpropagation.getGradient(theta, nnCostFunction.gettheta(), inputlayersize, hiddenlayersize, numlabels, X, y, lambda, alpha, iter, true);

//            theta1 = gradientDescentBackpropagation.getTheta1();
//            theta2 = gradientDescentBackpropagation.getTheta2();
            System.out.println("Test the decresing value of cost stored as feild in gradientDescentBackpropagation"+nnCostFunction.cost(gradTheta,inputlayersize,hiddenlayersize,
                    numlabels, X, y, lambda));
            /*****Gradient Descent End*****/
            AccuracyFeedforward accuracyFeedforward = new AccuracyFeedforward();
            int[][] pred = accuracyFeedforward.predict(X, gradientDescentBackpropagation.getTheta1(), gradientDescentBackpropagation.getTheta2(), 1);
//            double[][] pred = accuracyFeedforward.predictClass(gradientDescentBackpropagation.getPred(), 0);
//            double[][] pred = nnCostFunction.getPred();
            LogisticPredict logisticPredict = new LogisticPredict();
            logisticPredict.accuracy(pred, y);
            /***** Cost End *****/

        } catch (IOException e) {
            e.printStackTrace(); //given input file not found exception
        }
    }

    /*****theta argument counts equal to number of hidden layers *****/
    public void predict(){
        double[][] X = {{31,18.02,27.6,117.5,1013, 0.09489, 0.1036, 0.1086, 0.07055, 0.1865, 0.06333 ,0.6249, 1.89, 3.972, 71.55 ,0.004433, 0.01421, 0.03233, 0.009854, 0.01694 ,0.003495, 21.63 ,37.08, 139.7, 1436, 0.1195, 0.1926, 0.314, 0.117, 0.2677, 0.08113, 5, 5}};
        double[][] y = {{2}};
        int theta1Row = theta1.length;
        int theta1Col = theta1[0].length;

        int[] result = new int[theta1.length];

        /***
         * a1 input at first layer
         * a2 input at second layer
         * a3 output of the network

         * theta1 X a1
         * theta2 X a2

         * z2 output of first layer which is second layer(hidden layer); added a row of ones in z2 and changed it to a2
         * z3 output of second layer which is third layer(output layer); changed it to a3
         * ***/

        double[][] a1 = new double[X.length][X[0].length+1]; //input units
        /*******add column of ones in a1******/
        for(int i=0; i<X.length; i++){
            a1[i][0] = 1;
            for(int j=0; j<X[0].length; j++){
                a1[i][j+1] = X[i][j];
            }
        }

        Matrix matrix = new Matrix();

        double[][] a1Trans = matrix.transpose(a1);

        double[][] z2 = matrix.multMatrix(theta1, a1Trans);

        Sigmoid sigmoid = new Sigmoid();
        for(int i=0; i<theta1Row; i++){
            z2[i][0] = 1;
            for(int j=0; j<z2[0].length; j++){
               z2[i][j] = sigmoid.get(z2[i][j]);
            }
        }

        int z2Row = z2.length;
        int z2Col = z2[0].length;
        double[][] a2 = new double[z2Row+1][z2Col];
        /****add rows of ones in a2****/
        for(int i=0; i<z2Col; i++){
            a2[0][i] = 1;
        }
        for(int i=0; i<z2Row; i++){
            for(int j=0; j<z2Col; j++){
                a2[i+1][j] = z2[i][j];
            }
        }

        double[][] z3 = matrix.multMatrix(theta2, a2);

        int theta2Row = theta2.length;
        int theta2Col = theta2[0].length;

        int z3Col = z3[0].length;
        double[][] a3 = new double[theta2Row][z3Col];
        for(int i=0; i<theta2Row; i++){
            for(int j=0; j<z3Col; j++){
                a3[i][j] = sigmoid.get(z3[i][j]);
            }
        }

        double[][] transpose = matrix.transpose(a3);

        int transposeRow = transpose.length;
        int transposeCol = transpose[0].length;
        double[] max = new double[transposeRow];
        double[] predict = new double[transposeRow];
        double maxIndex = 0;
        int sumForMean = 0;

        /****Prediction Start: it works for any data inputs ****/
        for(int i=0; i<transposeRow; i++){
            max[i] = transpose[i][0];
            for(int j=1; j<transposeCol; j++){
                if(transpose[i][j]>max[i]){
                    max[i] = transpose[i][j];
                    maxIndex = j+1;
                    predict[i] = X[i][j];
                }
            }
            if(maxIndex==y[i][0]){
                sumForMean++;
            }
        }
        //here max[] stores the predicted values
        double accuracy = (sumForMean*100/transposeRow);
        System.out.println("Feedforward Neural Network accuracy: "+accuracy+"%");
        /****Prediction End: it works for any data inputs ****/
    }
}
