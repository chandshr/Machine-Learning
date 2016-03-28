package controllers.machineLearning.linear;

import controllers.machineLearning.all.Matrix;

/**
 * Created by chandani on 11/12/15.
 */
public class LinearGradientDescent {

    private double[][] theta;
    private double[] J_history;
    private double cost;

    public double getCost(){
        return this.cost;
    }

    /********get machineLearning.linear.LinearGradientDescent*******/
    /****data[][] is featureNormalized X***/
    public double[][] getGradient(double[][] theta, double[][] X, double[][] y, double alpha, int iter){
        int row = X.length;
        int col = X[0].length;

        double[] J_history = new double[iter];
        LinearCost linearCost = new LinearCost();
        Matrix matrix = new Matrix();
        double[][] diff = null;
        double[][] multXTheta;
        double[][] transDiff;
        double[][] multtransDiffandX;
        double[][] transposeOFmulttransDiffandX;
        double[][] mult;
        double x;
        cost = 10;
        for(int k=0; k<iter; k++){
            multXTheta = matrix.multMatrix(X, theta);
            diff = matrix.elementwiseOp(multXTheta, y, "-");
            transDiff = matrix.transpose(diff);
            multtransDiffandX = matrix.multMatrix(transDiff, X);
            x = (double) alpha/row;
            mult = matrix.matrixDivideorMultBy(matrix.transpose(multtransDiffandX), x, "*");
            theta = matrix.elementwiseOp(theta, mult, "-");
            cost = linearCost.getCost(X, y, theta);
            System.out.println("Decreasing cost: "+cost);
        }
        this.theta = theta;
//        this.J_history = J_history;
        return theta;
    }
}
