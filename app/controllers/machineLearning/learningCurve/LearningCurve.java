package controllers.machineLearning.learningCurve;

import controllers.machineLearning.all.*;
import controllers.machineLearning.linear_logistic.FeatureNormalize;
import controllers.machineLearning.linear_logistic.MapFeature;
import controllers.machineLearning.linear_logistic.RegressionInterface;


import java.util.HashMap;

/**
 * Created by shrestha on 1/11/2016.
 */
public class LearningCurve {
    private double[][] dataArr;

    private static int testRow;
    private static int crossValidRow;
    private static int trainRow;

    private FeatureNormalize trainfeatureNormalize = new FeatureNormalize();

    private double[][] xTrain;
    private double[][] yTrain;
    private double[][] xfeatureNormTrain;

    private double[][] xCross;
    private double[][] yCross;

    private double[][] xTest;
    private double[][] yTest;

    public void initializeTrainTestcrossValidData(String filepath, String jsonfilePath, String seperator, int colNo){
        TextToArr textToArr = new TextToArr();
        dataArr = textToArr.convertInputFiletoInputArr(filepath, seperator);
        OutputJson json = new OutputJson();
        json.createJsonFile(dataArr, jsonfilePath);

        int row = dataArr.length;
        int col = dataArr[0].length;

        testRow = (int) (.10*row);
        crossValidRow = testRow;
//        crossValidRow = 0;
        trainRow = row-2*testRow;

        /******* Train X and y start******/
        xTrain = new double[trainRow][col-1];
        yTrain = new double[trainRow][1];
        int a = 0;
        for(int i=0; i<trainRow; i++){
            yTrain[i][0] = dataArr[i][colNo-1];
            for (int j=0; j<col; j++){
                if(j!=colNo-1) {
                    double val = dataArr[i][j];
                    xTrain[i][a] = val;
                    a++;
                }
            }
            a=0;
        }
        /******* Train X and y end******/

        /******* Cross Validation X and y start******/
        a = 0;
        xCross = new double[crossValidRow][col-1];
        yCross = new double[crossValidRow][1];
        for(int i=0; i<crossValidRow; i++){
            yCross[i][0] = dataArr[i+trainRow][colNo-1];
            for (int j=0; j<col-1; j++){
                if(colNo-1!=j){
                    xCross[i][a] = dataArr[i+trainRow][j];
                    a++;
                }
            }
            a=0;
        }
        /******* Cross Validation X and y end******/

        /******* Test X and y start******/
        a = 0;
        xTest = new double[testRow][col-1];
        yTest = new double[testRow][1];
        for(int i=0; i<testRow; i++){
            yTest[i][0] = dataArr[i+trainRow+crossValidRow][colNo-1];
            for (int j=0; j<col-1; j++){
                if(colNo-1!=j){
                    xTest[i][a] = dataArr[i+trainRow+crossValidRow][j];
                    a++;
                }
            }
            a=0;
        }
        /******* Test X and y end******/

        //for machineLearning.logistic value at y[i][2]
//        /******* Train X and y start******/
//        int trainRow = trainData.length;
//        xTrain = new double[trainRow][col-1];
//        yTrain = new double[trainRow][1];
//        for(int i=0; i<trainRow; i++){
//            for (int j=0; j<col-1; j++){
//                if(j==2){
//                    yTrain[i][0] = trainData[i][j];
//                }else{
//                    xTrain[i][j] = trainData[i][j];
//                }
//            }
//        }
//        /******* Train X and y end******/
//
//        /******* Cross Validation X and y start******/
//        int crossValidRow = crossvalidationData.length;
//        xCross = new double[crossValidRow][col-1];
//        yCross = new double[crossValidRow][1];
//        for(int i=0; i<crossValidRow; i++){
//            for (int j=0; j<col-1; j++){
//                if(j==2){
//                    yCross[i][0] = crossvalidationData[i][j];
//                }else{
//                    xCross[i][j] = crossvalidationData[i][j];
//                }
//            }
//        }
//        /******* Cross Validation X and y end******/
//
//        /******* Test X and y start******/
//        int testRow = testData.length;
//        xTest = new double[testRow][col-1];
//        yTest = new double[testRow][1];
//        for(int i=0; i<testRow; i++){
//            for (int j=0; j<col-1; j++){
//                if(j==2){
//                    yTest[i][0] = testData[i][j];
//                }else{
//                    xTest[i][j] = testData[i][j];
//                }
//            }
//        }
        /******* Test X and y end******/

        /******* Train Data feature normalization start*******/
//        double[][] xtrainfeatureNormData = new double[row][col];
//        xtrainfeatureNormData = trainfeatureNormalize.getFeatureNormalize(xTrain);
//        xTrain = xtrainfeatureNormData;
//
//        Matrix matrix = new Matrix();
//        xTrain = matrix.addColOfOnes(xTrain);
//
//
////      trainX is updated with featureNormalization and 1 is added infront of X
//        for(int i=0; i<trainRow; i++){
//            xTrain[i][0] = 1;
//            for (int j=1; j<col; j++){
//                xTrain[i][j] = xtrainfeatureNormData[i][j-1];
//            }
//        }
//      this returns a normalized and added 1's infront of trainX
        /******* Train Data feature normalization end*******/
    }

    public HashMap<Double, Double> getCostLamdaHash(double[][] X, double[][] y, double[][] theta, double startLamda, double endLambda, RegressionInterface regressionInterface){
        HashMap<Double, Double> costLambdaHash = new HashMap<Double, Double>();
        double cost;
        double lambda = startLamda;//0.64; E303
        cost = regressionInterface.getCost(X, y, theta);
        costLambdaHash.put(0.0, cost); //cost at lamda = 0
        for(lambda=startLamda; lambda<=endLambda; lambda=lambda*2){
            cost = regressionInterface.getCost(X, y, theta);
            costLambdaHash.put(lambda, cost);
        }
        return costLambdaHash;
    }

    public HashMap<Integer, Double> getCostDegreeHash(double[][] X, double[][] y, double[][] theta, int degree, RegressionInterface regressionInterface){
        HashMap<Integer, Double> costDegreeHash = new HashMap<Integer, Double>();
        double cost;
//        double lambda = 0.128; //optimum lambda 0.128
//        FeatureNormalize featureNormalize = new FeatureNormalize();
//        dataArr = featureNormalize.getFeatureNormalize(dataArr);
        cost = regressionInterface.getCost(X, y, theta);
        costDegreeHash.put(1, cost);
        MapFeature mapFeature = new MapFeature();
        for(int i=2; i<=degree; i++){
            dataArr = mapFeature.mapFeature(dataArr, i);
            cost = regressionInterface.getCost(X, y, theta);
            costDegreeHash.put(i, cost);
        }
        return costDegreeHash;
    }

    public HashMap<Integer, Double> getCostTraincountHash(double[][] X, double[][] y, double[][] theta, int traincount, RegressionInterface regressionInterface){
        HashMap<Integer, Double> costTraincountHash = new HashMap<Integer, Double>();
        int col = dataArr[0].length;
        double cost;
//        double lambda = 0.128;
        double[][] incCountInput; //increases input by adding a row from original input e.g at first it has one row of example, then it has two row, then three row and consecutively
        for(int i=0; i<traincount; i++){ //i=0; input with one data point
            incCountInput = new double[i+1][];
            for(int j=0; j<=i; j++){
                incCountInput[j] = dataArr[j];
            }
            cost = regressionInterface.getCost(X, y, theta);//lambda = 0
            costTraincountHash.put(i+1, cost);
        }
        return costTraincountHash;
    }

    public FeatureNormalize getTrainfeatureNormalize() {
        return trainfeatureNormalize;
    }

    public void setTrainfeatureNormalize(FeatureNormalize trainfeatureNormalize) {
        this.trainfeatureNormalize = trainfeatureNormalize;
    }

    public double[][] getxTrain() {
        return xTrain;
    }

    public void setxTrain(double[][] xTrain) {
        this.xTrain = xTrain;
    }

    public double[][] getyTrain() {
        return yTrain;
    }

    public void setyTrain(double[][] yTrain) {
        this.yTrain = yTrain;
    }

    public double[][] getXfeatureNormTrain() {
        return xfeatureNormTrain;
    }

    public void setXfeatureNormTrain(double[][] xfeatureNormTrain) {
        this.xfeatureNormTrain = xfeatureNormTrain;
    }

    public double[][] getxTest() {
        return xTest;
    }

    public void setxTest(double[][] xTest) {
        this.xTest = xTest;
    }

    public double[][] getyTest() {
        return yTest;
    }

    public void setyTest(double[][] yTest) {
        this.yTest = yTest;
    }

    public double[][] getxCross() {
        return xCross;
    }

    public void setxCross(double[][] xCross) {
        this.xCross = xCross;
    }

    public double[][] getyCross() {
        return yCross;
    }

    public void setyCross(double[][] yCross) {
        this.yCross = yCross;
    }

    public static int getTestRow() {
        return testRow;
    }

    public static void setTestRow(int testRow) {
        LearningCurve.testRow = testRow;
    }

    public static int getCrossValidRow() {
        return crossValidRow;
    }

    public static void setCrossValidRow(int crossValidRow) {
        LearningCurve.crossValidRow = crossValidRow;
    }

    public static int getTrainRow() {
        return trainRow;
    }

    public static void setTrainRow(int trainRow) {
        LearningCurve.trainRow = trainRow;
    }

}
