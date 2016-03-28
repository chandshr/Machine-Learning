package controllers.machineLearning.all;

/**
 * Created by shrestha on 1/6/2016.
 */
public class PlotDecisionBoundary {

    /**
     * returns the extreme points of a line (decision boundary)
     * @param theta
     * @param X
     * @param y
     * @return
     */
    public double[][] getLinePoints(double[][] theta, double[][] X, double[][] y){
        Matrix matrix = new Matrix();
        double[] plot_x = new double[2];
        double[] plot_y = new double[2];
        double[][] result = new double[2][2];

        double[] XcolInput = matrix.getColArr(X, 0); //0th col element

        plot_x[0] = matrix.oneDimensionalOp(XcolInput, "min") -2;
        plot_x[1] = matrix.oneDimensionalOp(XcolInput, "max") +2;

        for(int i=0; i<plot_x.length; i++){
            plot_y[i] = (-1/theta[2][0])*(theta[1][0]*plot_x[i]+theta[0][0]);
            result[i][0] = plot_x[i];
            result[i][1] = plot_y[i];
        }
        return result;
    }
}
