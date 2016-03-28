package controllers.machineLearning.linear;

/**
 * Created by shrestha on 11/18/2015.
 */
public class LinearRegGradient {

    public double[][] getGradient(double lambda, double alpha, int row, double[][] theta){
        int col = theta[0].length;
        double thetaSum = 0;
        double[][] thetaFiltered = new double[1][col];
        for(int i=1; i<col; i++){
            thetaFiltered[0][i] = (lambda/row)*theta[0][i];
            theta[0][i] = theta[0][i] + thetaFiltered[0][i];
        }
        return theta;
    }
}
