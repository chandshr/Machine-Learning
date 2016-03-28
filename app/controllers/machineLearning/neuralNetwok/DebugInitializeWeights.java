package controllers.machineLearning.neuralNetwok;

/**
 * Created by shrestha on 12/22/2015.
 */
public class DebugInitializeWeights {

    public double[][] createMatrix(int row, int col){
        double[][] W = new double[row][col+1];
        int count = 1;
        for(int i=0; i<col+1; i++){
            for(int j=0; j<row; j++){
                W[j][i] = Math.sin(count)/10;
                count++;
            }
        }
        return W;
    }
}
