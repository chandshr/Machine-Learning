package controllers.machineLearning.neuralNetwok;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.all.Sigmoid;

/**
 * Created by shrestha on 1/21/2016.
 */
public class AccuracyFeedforward {

    public void displayAccuracy(double[][] a3, double[][] X, double[][] y){
        Matrix matrix = new Matrix();
        /**
         * Accuracy Calculation
         */
        double[][] transpose = matrix.transpose(a3);

        int transposeRow = transpose.length;
        int transposeCol = transpose[0].length;
        double[] max = new double[transposeRow];
        double[] predict = new double[transposeRow];
        double maxIndex = 1;
        int sumForMean = 0;

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
    }

    public int[][] predict(double[][] X, double[][] theta1, double[][] theta2, int startClass){
        int row = X.length;
        Sigmoid sigmoid = new Sigmoid();
        Matrix matrix = new Matrix();
        X = matrix.addColOfOnes(X);
        double[][] transTheta1 = matrix.transpose(theta1);
        double[][] multTransTheta1X = matrix.multMatrix(X, transTheta1);
        double[][] h1 = sigmoid.getSigmoidArr(multTransTheta1X);
        h1 = matrix.addColOfOnes(h1);
        double[][] transTheta2 = matrix.transpose(theta2);
        double[][] multTheta2H1 = matrix.multMatrix(h1, transTheta2);
        double[][] h2 = sigmoid.getSigmoidArr(multTheta2H1);
        int[][] predClass = new int[row][1];
        double[][] predH = new double[row][1];
        for(int i=0; i<h2.length; i++){
            int maxClass = 0+startClass; //considering class starting from 1;
            double max = h2[i][0];
            for(int j=1; j<h2[0].length; j++){
                if(h2[i][j]>max){
                    max  = h2[i][j];
                    maxClass = j+startClass;
                    predH[i][0] = h2[i][j];
                }
            }
            predClass[i][0] = maxClass;
        }
//        return h2;
        return predClass;
    }

    public double[][] predictClass(double[][] h, int startClass){
        double[][] predH = new double[h.length][1];
        double[][] predClass = new double[h.length][1];
        for(int i=0; i<h.length; i++){
            double maxClass = 0+startClass; //considering class starting from 1;
            double max = h[i][0];
            for(int j=1; j<h[0].length; j++){
                if(h[i][j]>max){
                    max  = h[i][j];
                    maxClass = j+startClass;
                    predH[i][0] = h[i][j];
                }
            }
            predClass[i][0] = maxClass;
        }
        return predClass;
    }
}
