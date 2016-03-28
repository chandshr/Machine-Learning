package controllers.machineLearning.neuralNetwok;

import controllers.machineLearning.all.TextToArr;
import controllers.machineLearning.linear_logistic.FeatureNormalize;

/**
 * Created by shrestha on 12/18/2015.
 */
public class Backwardpropopagation {
    public void get(){
        NNCostFunction nnCostFunction = new NNCostFunction();
        TextToArr textToArr = new TextToArr();
//        try {
//            File currentDirectory = new File(new File(".").getAbsolutePath());
            String slash = "/";
//            String xPath = currentDirectory.getCanonicalPath()+"/src/main/resources/data/breast/breastX.txt";
//            String yPath = currentDirectory.getCanonicalPath()+slash+"/src/main/resources/data/breast/breastY.txt";
            String filepath = "/src/main/resources/data/bike/hour.csv";
            String jsonfilePath = "/src/main/resources/data/json/linearInput.json";

            /***** 1. Learning Curve Start ******/
//            LearningCurve learningCurve = new LearningCurve();
//            learningCurve.initializeTrainTestcrossValidData(filepath, jsonfilePath);//initialize feature normalized train, test and cross validation data
//            double[][] xTrain =learningCurve.getxTrain();
//            double[][] yTrain = learningCurve.getyTrain();
//            double[][] xCross =learningCurve.getxCross();
//            double[][] yCross = learningCurve.getyCross();
//            double[][] xTest = learningCurve.getxTest();
//            double[][] yTest = learningCurve.getyTest();
//            FeatureNormalize trainfeatureNormalize = learningCurve.getTrainfeatureNormalize();
            /****** Learning Curve End *****/
//            String strSeperator = "\\s+";
            String strSeperator = ",";
//            double[][] X = xTrain;
//            double[][] y = yTrain;
//            ArrayList<ArrayList<Double>> xdata = textToArr.convert(xPath, strSeperator);
//            ArrayList<ArrayList<Double>> ydata = textToArr.convert(yPath, strSeperator);

        double[][] dataArr = textToArr.convertInputFiletoInputArr(filepath, strSeperator);
        double[][] X = new double[dataArr.length][dataArr[0].length-1];
        double[][] y = new double[dataArr.length][1];
        for(int i=0; i<dataArr.length; i++){
            for(int j=0; j<dataArr[0].length-1; j++){
                X[i][j] = dataArr[i][j];
            }
            y[i][0] = dataArr[i][dataArr[0].length-1];
        }

//            double[][] X = textToArr.datatoArr(xdata);
//            double[][] y = textToArr.datatoArr(ydata);

            FeatureNormalize featureNormalize = new FeatureNormalize();
            X = featureNormalize.getFeatureNormalize(X);
            DebugInitializeWeights debugInitializeWeights = new DebugInitializeWeights();
            int inputLayerSize = X[0].length;
            int hiddenLayerSize = inputLayerSize/2;
//            int outputlayersize = 10;
            int outputlayersize = 2;
            double lambda = 0.1;
            double[][] theta1 = debugInitializeWeights.createMatrix(hiddenLayerSize, inputLayerSize);
            double[][] theta2 = debugInitializeWeights.createMatrix(2, hiddenLayerSize);
            double[][] theta = combineTheta(theta1, theta2);

            /******Check Neural Network Start*****/
//            CheckNNGradients checkNNGradients = new CheckNNGradients();
//            checkNNGradients.checkNeuralNetwork(lambda,inputLayerSize, hiddenLayerSize, outputlayersize, X, y, theta1, theta2, theta, false);
//            checkNNGradients.checkNeuralNetwork(lambda);
            /******Check Neural Network End*****/

            /*****Gradient Descent Start*****/
            double alpha = 0.01;
            int iter = 8;
            GradientDescentBackpropagation gradientDescentBackpropagation = new GradientDescentBackpropagation();
            theta = gradientDescentBackpropagation.getGradient(theta, nnCostFunction.getDelta(),inputLayerSize,hiddenLayerSize,
                    outputlayersize, X, y, lambda, alpha, iter, false);

//            //Feb 2 start
//            double[] J_history = new double[iter];
//            for(int i=0; i<iter; i++){
//                J_history[i] = nnCostFunction.cost(theta, inputLayerSize, hiddenLayerSize, outputlayersize, X, y, lambda);
//                theta = nnCostFunction.gettheta();
//            }
//            //Feb 2 end
            /*****Gradient Descent End*****/
            double cost = nnCostFunction.cost(theta, inputLayerSize, hiddenLayerSize, outputlayersize, X, y, lambda);
            System.out.println("Neural Network cost: "+cost);

            /******Gradient Descent Start*****/
//            LogisticGradientDescent gradientDescent = new LogisticGradientDescent();
//            Matrix matrix = new Matrix();
//            double[][] initalTheta = {{0}, {0}, {0}};
//            double alpha = 0.001;
//            int iter = 4000;
//            X = matrix.addColOfOnes(X);
//
//            double[][] gradTheta = gradientDescent.getGradientDescent(matrix.transpose(theta1), X, y, iter, alpha);
//            double gradCost = NNCostFunction.cost(gradTheta, inputLayerSize, hiddenLayerSize, outputlayersize, X, y, lambda);
//            System.out.println("Grad cost is "+gradCost+" normal cost is "+cost);
            /******Gradient Descent End*****/
//        }catch (IOException e){
//            e.printStackTrace(); //given input file not found exception
//        }
    }

    public double[][] combineTheta(double[][] theta1, double[][] theta2){
        int a = theta1.length*(theta1[0].length)+theta2.length*(theta2[0].length);
        double[][] theta = new double[theta1.length*(theta1[0].length)+theta2.length*(theta2[0].length)][1];
        int count = 0;
        for(int i=0; i<theta1[0].length; i++){
            for(int j=0; j<theta1.length; j++){
                theta[count][0] = theta1[j][i];
//                System.out.println(theta[count][0]);
                count++;
            }
        }
        for(int i=0; i<theta2[0].length; i++){
            for(int j=0; j<theta2.length; j++){
                theta[count][0] = theta2[j][i];
                count++;
            }
        }
        return theta;
    }
}
