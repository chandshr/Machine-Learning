package controllers.machineLearning.neuralNetwok;

/**
 * Created by shrestha on 12/12/2015.
 */
public class RandInitializeWeights {

    public double[][] getRandomWeights(int outUnit, int inUnit, double epsilon){
        double[][] theta = new double[outUnit][inUnit+1];
        for(int i=0; i<outUnit; i++){
            for(int j=0; j<inUnit; j++){
                theta[i][j] = Math.random()*2*epsilon - epsilon;
            }
        }
        return theta;
    }

    public double[][] randomWeight(int row, int col, double epsilon){
        double[][] W = new double[row][col+1];
        int count = 1;
        for(int i=0; i<col+1; i++){
            for(int j=0; j<row; j++){
                W[j][i] = Math.random()*2*epsilon - epsilon;
                count++;
            }
        }
        return W;
    }
}
