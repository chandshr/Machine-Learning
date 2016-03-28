package controllers.machineLearning.linear_logistic;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.logistic.LogisticCost;

/**
 * Created by shrestha on 11/18/2015.
 */
public class RegularizedGradient {

    private double regularizedCost;
    public double[][] getGradient(double[][] theta, double[][] X, double[][] y, double lambda, String algorithm){
//        int col = theta[0].length;
//        double thetaSum = 0;
//        double[][] thetaFiltered = new double[1][col];
//        for(int i=1; i<col; i++){
//            thetaFiltered[0][i] = (lambda/row)*theta[0][i];
//            theta[0][i] = theta[0][i] + thetaFiltered[0][i];
//        }
//        return theta;
        Matrix matrix = new Matrix();
        int thetarow = theta.length;
        int row = y.length;
        double cost;
        double[][] grad = theta;
        if(algorithm== "machineLearning/logistic"){
            double[][] thetaFiltered = new double[thetarow][1];
            thetaFiltered = theta;
            thetaFiltered[0][0] = 0;
            LogisticCost logisticCost = new LogisticCost();
            cost = logisticCost.getCost(X, y, theta);
            grad = logisticCost.getGrad();
            double[][] costmult = matrix.multMatrix(matrix.transpose(thetaFiltered),thetaFiltered);
            double a = (double) lambda/(2*row);
            double[][] costmultAndlambda = matrix.matrixDivideorMultBy(costmult, a, "*");
            cost = cost+costmultAndlambda[0][0];
            this.regularizedCost = cost;

            //calculate gradient descent
            double x = (double) lambda/row;
            double[][] mult = matrix.matrixDivideorMultBy(thetaFiltered, x, "*");
            grad = matrix.elementwiseOp(grad, mult, "+");
        }
        return grad;
    }

    public double getRegularizedCost() {
        return regularizedCost;
    }

    public void setRegularizedCost(double regularizedCost) {
        this.regularizedCost = regularizedCost;
    }
}
