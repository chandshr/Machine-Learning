package controllers.machineLearning.logistic;

/**
 * Created by shrestha on 11/18/2015.
 */
public class LogisticRegCost {
    public double regCost(double[][]X, double[][]y, double[][]theta, double lambda){
        LogisticCost logisticCost = new LogisticCost();
        double cost = logisticCost.getCost(theta, X, y);
        int row = X.length;
        int col = theta[0].length;
        double thetaSum = 0;
        for(int i=1; i<col; i++){
            thetaSum += theta[0][i];
        }
        double regCost = cost + lambda/(2*row)*thetaSum;
        return regCost;
    }
}
