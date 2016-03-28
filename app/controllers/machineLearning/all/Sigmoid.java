package controllers.machineLearning.all;

/**
 * Created by shrestha on 11/16/2015.
 */
public class Sigmoid {

    public double get(double z){
        double g = 1/(1+Math.exp(-z));
        return g;
    }

    public double[][] getSigmoidArr(double[][] input){
        int row = input.length;
        int col = input[0].length;
        double[][] sigmoidArr = new double[row][col];
        for(int i=0; i<input.length; i++){
            for(int j=0; j<input[0].length; j++){
                sigmoidArr[i][j] = get(input[i][j]);
            }
        }
        return sigmoidArr;
    }

    public double[][] sigmoidGradient(double[][] input){
        int row = input.length;
        int col = input[0].length;
        double[][] result = new double[row][col];
        double[][] temp = getSigmoidArr(input);
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                result[i][j] = temp[i][j]*(1-temp[i][j]);
            }
        }
        return result;
    }
}
