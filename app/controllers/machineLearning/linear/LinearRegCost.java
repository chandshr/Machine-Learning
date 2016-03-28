package controllers.machineLearning.linear;

/**
 * Created by shrestha on 11/18/2015.
 */
public class LinearRegCost {

    public double regCost(double[][]X, double[][]y, double[][]theta, double lamda){
        LinearCost linearCost = new LinearCost();
        double cost = linearCost.getCost(X, y, theta);
        double regCost = 0;
        double thetaSum = 0;
        int row = X.length;
        int col = theta[0].length;
        for(int i=1; i<col; i++){
            thetaSum += (theta[0][i]*theta[0][i]);
        }
        regCost = cost + (lamda/(2*row))*thetaSum;
        return regCost;
    }
}
