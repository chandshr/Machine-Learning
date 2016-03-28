package controllers.machineLearning.linear_logistic;

/**
 * Created by shresc2 on 11/12/2015.
 */

/****
 * Feature Normalization is used when the value of x is not in the given range: -3<x<3(larger range); -1/3,x,1/3(smaller range); -1<x<1
 * Follow the lecture of Andrew to have clear understanding: https://class.coursera.org/ml-003/lecture/21
 * ****/
//TODO: remove unnecessary class variables and function
public class FeatureNormalize {
    private double[] mean;
    private double[] std; //larger-smaller so, if larger == smaller it returns NAN
    private double[] range;
    private double[][] featureNormalize;

    /****This function called in getFeatureNormalize(data)***/
    public void xNormVar(double[][] data){
        /*******this function is dealing with calculating with X_norm variables, X has only col-1 columns******/
        int row = data.length;
        int col = data[0].length;
        double[] mean = new double[col];
        double[] max = new double[col];
        double[] min = new double[col];
        double[] range = new double[col];
        double[] std = new double[col];
        double diff;

        /*initialization*/
        for(int j=0; j<col; j++){
             mean[j] = data[0][j];
             max[j] = data[0][j];
             min[j] = data[0][j];
            //standard deviation
            diff = data[0][j] - mean[j];
        }

        for(int i=1; i<row; i++){
            for(int j=0; j<col; j++){
                mean[j] = mean[j]+data[i][j];
                double ma = max[j];
                double mi = min[j];
                double da = data[i][j];
                max[j]  = max[j] > data[i][j] ? max[j] : data[i][j];
                min[j]  = min[j] < data[i][j] ? min[j] : data[i][j];
                //std
                diff = data[i][j] - mean[j];
            }
        }

        for(int j=0; j<col; j++){
            mean[j] = mean[j]/row;
            range[j] = max[j]-min[j];
            std[j] = (double)Math.sqrt(std[j]/(row));
        }


        this.mean = mean;
        this.range = range;
        this.std(data);
    }

    public void std(double[][] data){
        int row = data.length;
        int col = data[0].length;
        double[] std = new double[col];
        double[] mean = this.mean;
        double diff;
        /*initialization*/
        for(int j=0; j<col; j++){
            diff = data[0][j] - mean[j];
            std[j] = (double)Math.pow(diff, 2);
        }
        for(int i=1; i<row; i++){
            for(int j=0; j<col; j++){
                diff = data[i][j] - mean[j];
                std[j] += (double)Math.pow(diff, 2);
            }
        }
        for(int j=0; j<col&&(row-1!=0); j++){
            std[j] = (double)Math.sqrt(std[j]/(row-1));
        }
        this.std = std;
    }

    public double[][] getFeatureNormalize(double[][] X){
        xNormVar(X);
        int row = X.length;
        int col = X[0].length;
        double[][] normalizedX = new double[row][col];
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                double diff = X[i][j]-this.mean[j];
                if(std[j]!=0)
                    normalizedX[i][j] = diff/std[j];
                else
                    normalizedX[i][j] = diff;
            }
        }
        return normalizedX;
    }

    public double[][] predFeatureNormalize(double[][] X, double[] std, double[] mean){
        int row = X.length;
        int col = X[0].length;
        double[][] normalizedX = new double[row][col];
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                double diff = X[i][j]-mean[j];
                if(std[j]!=0)
                    normalizedX[i][j] = diff/std[j];
                else
                    normalizedX[i][j] = diff;
            }
        }
        return normalizedX;
    }

    public double[] getMean() {
        return mean;
    }

    public void setMean(double[] mean) {
        this.mean = mean;
    }

    public double[] getStd() {
        return std;
    }

    public void setStd(double[] std) {
        this.std = std;
    }
}
