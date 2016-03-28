package controllers.machineLearning.logistic;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.all.Sigmoid;
import controllers.machineLearning.linear_logistic.RegressionInterface;

/**
 * Created by shrestha on 11/16/2015.
 */
public class LogisticCost implements RegressionInterface{

    //del start
    private double[][] grad;
    public double[][] getGrad() {
        return grad;
    }

    public void setGrad(double[][] grad) {
        this.grad = grad;
    }
    //del end

    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }

    private double cost;

    public double getCost(double[][] X, double[][] y, double[][] theta){
        int row = X.length;
        int col = X[0].length;

        this.grad = theta;

        Matrix matrix = new Matrix();
        Sigmoid sigmoid = new Sigmoid();

        double[][] XthetaMult = matrix.multMatrix(X, theta);
        int XthetaMultRow = XthetaMult.length;
        int XthetaMultCol = XthetaMult[0].length;
//        double[][] h = new double[XthetaMultRow][XthetaMultCol];
        double[][] h = new double[XthetaMultRow][1];

        for(int i=0; i<XthetaMultRow; i++){
//            for(int j=0; j<XthetaMultCol; j++){
//                h[i][j] = sigmoid.get(XthetaMult[i][j]);
//            }
            double a = XthetaMult[i][0];
            h[i][0] = sigmoid.get(XthetaMult[i][0]);
        }

        /**********cost start************/
        double costPos = 0;
        double costNeg = 0;
        int theta_col = theta[0].length;
        for(int i=0; i<row; i++){
            costPos += (-y[i][0]*Math.log(h[i][0]));
            costNeg += (1-y[i][0])*Math.log(1-h[i][0]);
        }
        /**** cost end ****/

        /********Gradient Descent start*********/
        double[][] diff = matrix.elementwiseOp(h, y, "-");
        double[][] transposeOfX = matrix.transpose(X);
        double[][] multOfXtransDiff = matrix.multMatrix(transposeOfX, diff);
        this.grad = new double[multOfXtransDiff.length][1];
        double x = (double) 1/row;
        this.grad = matrix.matrixDivideorMultBy(multOfXtransDiff, x, "*");
        /********Gradient Descent end*********/

        double J = (costPos-costNeg)/row;
        this.cost = J;
        return J;
    }
}
