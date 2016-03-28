package controllers.machineLearning.logistic;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.all.Sigmoid;

import java.util.Arrays;

/**
 * Created by shrestha on 11/17/2015.
 */
public class LogisticPredict {
    public double probability(double[][] theta, double[][] input){
        Sigmoid sigmoid = new Sigmoid();
        Matrix matrix = new Matrix();

        double[][] inputThetaMult = matrix.multMatrix(input, theta); //it returns a single element
        double prob = sigmoid.get(inputThetaMult[0][0]);
        return prob;
    }

    public int[][] predict(double[][] theta, double[][] X){
        Matrix matrix = new Matrix();
        double[][] multXTheta = matrix.multMatrix(X, theta);
        int row = multXTheta.length;
        int col = multXTheta[0].length;
        int[][] pred = new int[row][1];
        Sigmoid sigmoid = new Sigmoid();
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                double sig = sigmoid.get(multXTheta[i][j]);
                if(sig>=0.5){
                    pred[i][0] = 1;
                }else {
                    pred[i][0] = 0;
                }
            }
        }
        return pred;
    }

    public double[][] predictMultiClass(double[][] theta, double[][] X, int value){
        Matrix matrix = new Matrix();
        double[][] multXTheta = matrix.multMatrix(X, theta);
        int row = multXTheta.length;
        int col = multXTheta[0].length;
        double[][] pred = new double[row][1];
        Sigmoid sigmoid = new Sigmoid();
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                double sig = sigmoid.get(multXTheta[i][j]);
                pred[i][0] = sig;
            }
        }
        return pred;
    }

    public double accuracy(int[][] pred, double[][] y){
        int row = pred.length;
        double sum = 0;
        int counterSum = 0;
        int benign = 0; //2
        int malignant = 0; //4
        for(int i=0; i<row; i++){
            if(pred[i][0]==y[i][0]){
                sum++;
//                System.out.println(i+"th value of pred: "+pred[i][0]+" y: "+y[i][0]);
                if(y[i][0]==0){
                    benign++;
                }else{
                    malignant++;
                }
            }
          else{
                counterSum++;
                System.out.println(i+"th value of FALSE pred: "+pred[i][0]+" y: "+y[i][0]);
            }
        }
        System.out.println("Counter prediction: "+counterSum);
        System.out.println("Benign: "+benign);
        System.out.println("Malignant: "+malignant);
        double mean = sum/row;
        double accPercent = mean*100;
        System.out.println("Accuracy: "+accPercent);
        return accPercent;
    }
}
