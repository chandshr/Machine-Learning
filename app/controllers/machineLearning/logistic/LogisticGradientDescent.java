package controllers.machineLearning.logistic;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.all.Sigmoid;

/**
 * Created by shrestha on 12/23/2015.
 * this is for machineLearning.logistic RegressionInterface as it uses sigmoid, used in machineLearning.logistic regression
 * in linearRegression it doesn't use sigmoid
 */
public class LogisticGradientDescent {

    private double[] J_history;
    private double[][] theta;
    private double cost;

    public double[][] getGradientDescent(double[][] X, double[][] y, int iter, double alpha){
        this.theta = new double[X[0].length][1];
        this.J_history = new double[iter];

        Matrix matrix = new Matrix();
        Sigmoid sigmoid = new Sigmoid();

        double[][] diff = new double[y.length][1];
        int row = X.length;

        cost = 10;
        double[][] h = new double[row][1];
        for(int k=0; k<iter; k++){
            double[][] XthetaMult = matrix.multMatrix(X, theta);
            int XthetaMultRow = XthetaMult.length;


            for(int i=0; i<XthetaMultRow; i++){
                h[i][0] = sigmoid.get(XthetaMult[i][0]);
            }
            diff = matrix.elementwiseOp(h, y, "-");
            double[][] transposeOfX = matrix.transpose(X);
            double[][] multOfXtransDiff = matrix.multMatrix(matrix.transpose(diff), X);
            double x = (double) alpha/row;
            double[][] mult = matrix.matrixDivideorMultBy(matrix.transpose(multOfXtransDiff), x, "*");
            theta = matrix.elementwiseOp(theta, mult, "-");
            LogisticCost logisticCost = new LogisticCost();
            this.cost = logisticCost.getCost(X, y, theta);
//            this.J_history[k] = cost;
            System.out.println("Decreasing Cost: "+cost);
        }
        return theta;
    }
    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }
    public double[] getJhistory(){
        return this.J_history;
    }
}
