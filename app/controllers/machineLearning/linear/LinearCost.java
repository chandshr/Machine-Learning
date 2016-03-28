package controllers.machineLearning.linear;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.linear_logistic.RegressionInterface;

/**
 * Created by chandani on 11/13/15.
 */
public class LinearCost implements RegressionInterface {
    public double getCost(double[][]X, double[][]y, double[][]theta){
        int row = X.length;
        int col = X[0].length;

//        double cost = 0;
//        double[][] hyposumdiff = new double[row][1];
//        double sum = 0;
//        for(int i=0; i<row; i++){
//            double hyposum = 0;
//            for(int j=0; j<col; j++){
//                hyposum += theta[0][j]*X[i][j];
//            }
//            hyposumdiff[i][0] = hyposum-y[i][0];
//            double pow = Math.pow(hyposumdiff[i][0], 2);
//            sum += Math.pow(hyposumdiff[i][0], 2);
//        }
//        cost = sum/(2*row);
//        return cost;
        Matrix matrix = new Matrix();
        double[][] multThetaX = matrix.multMatrix(X, theta);
        double[][] diff = matrix.elementwiseOp(multThetaX, y, "-");
        double[][] transposediff = matrix.transpose(diff);
        double[][] mult = matrix.multMatrix(transposediff, diff);
        double x = (double) 2*row;
        double[][] cost = matrix.matrixDivideorMultBy(mult, x, "/");
        return cost[0][0];
    }
}
