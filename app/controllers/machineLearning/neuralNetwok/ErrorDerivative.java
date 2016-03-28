package controllers.machineLearning.neuralNetwok;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.all.Sigmoid;

/**
 * Created by shrestha on 2/7/2016.
 */
public class ErrorDerivative {

    private static double[][] outputDelta;
    private static double[][] inputDelta;

    public static double[][] getErrorDerivative(double[][] target, double[][] output){
        double[][] result = new double[target.length][1];
        for(int i=0; i<target.length; i++){
            result[i][0] = -(target[i][0]-output[i][0])*(1-output[i][0])*output[i][0];
        }
        return result;
    }

    public static void setOutputDelta(double[][] target, double[][] output, int delta2Size){
        outputDelta = new double[output.length][1];
        for(int i=0; i<output.length; i++){
            outputDelta[i][0] = (target[i][0] - output[i][0])*output[i][0]*(1-output[i][0]);
        }
    }

    public static void setInputDelta(double[][] target, double[][] output, double[][] X, double[][] theta2){
        Matrix matrix = new Matrix();
        inputDelta = new double[theta2.length][1];
        for(int i=0; i<theta2.length; i++){
            double ithRowXcost = theta2[i][0];
            for(int j=0; j<X[0].length; j++){
                ithRowXcost += outputDelta[i][j]*theta2[i][j];
            }
            inputDelta[i][0] = ithRowXcost*output[i][0]*(1-output[i][0]);
        }
    }

    public static double[][] getErrorDerivative(double[][] target, double[][] output, double[][] X, double[][] theta1, int delta2Size){
        setInputDelta(target, output, X, theta1);
        setOutputDelta(target, output, delta2Size);
        Backwardpropopagation backwardpropopagation = new Backwardpropopagation();
        double[][] errorDelta = backwardpropopagation.combineTheta(inputDelta, outputDelta);
        return errorDelta;
    }

    public double[][] getErrorDerivative(double[][] theta1, double[][] theta2, double[][] X, double[][] y, double[][] h, boolean multiclass, int numlabels){
        inputDelta = new double[theta1.length][theta1[0].length];
        outputDelta = new double[theta2.length][theta2[0].length];

        double[][] a1;
        double[][] a2;
        double[][] z2;
        double[][] a3 = h.clone();
        int Xrow = X.length;
        int yCol = y[0].length;
        int yRow = y.length;

        Matrix matrix = new Matrix();
        Sigmoid sigmoid = new Sigmoid();

        double[][] Y = new double[Xrow][numlabels]; //Y in octave backpropagation
        if(multiclass){
            int count = 0;
            for(int i=0; i<yCol; i++){
                for(int j=0; j<yRow; j++){
                    int yRowWiseVal = (int)y[j][i]-1; //if it has class 1, 2, ...
//                int yRowWiseVal = (int)y[j][i]; // if it has a class 0 use this
//                    Y[count][yRowWiseVal] = yRowWiseVal+1;
                    Y[count][yRowWiseVal] = 1;
                    count++;
                }
            }
        }else{
            Y = y;
        }

        for(int i=0; i<Xrow; i++){
            /**step1 start**/
//            double[] b = X[i];
            double[][] a = matrix.transpose(X[i]);
            a1 = matrix.addOneRowOfOnes(a);
//
            z2 = matrix.multMatrix(theta1, a1);
            a2 = matrix.addOneRowOfOnes(sigmoid.getSigmoidArr(z2));
//
            double[][] z3 = matrix.multMatrix(theta2, a2);
            a3 = sigmoid.getSigmoidArr(z3);
            /**step1 end**/

            /*** step 2 start ***/
            double[][] yt = matrix.transpose(Y[i]); //for multiclass
            double[][] d3 = matrix.elementwiseOp(a3, yt, "-");

            outputDelta[i][0] = matrix.elementwiseOp(d3, sigmoid.sigmoidGradient(d3), "*")[0][0];

            double ithRowXcost = theta2[i][0];
            for(int j=0; j<theta2.length; j++){
                ithRowXcost += outputDelta[i][j]*theta2[i][j];
            }
            inputDelta[i][0] =ithRowXcost*sigmoid.sigmoidGradient(a2)[0][0];
            /*** step 2 end ***/

            /*** Step 3 Start ***/
//            double[][] theta2Filtered = new double[theta2.length][theta2[0].length-1];
//            theta2Filtered = matrix.copy(theta2Filtered, theta2Filtered, 1);
//            double[][] multTheta2FiltransAndD3 = matrix.multMatrix(matrix.transpose(theta2Filtered), d3);
//
//            double[][] sigmoidGradientOfZ2 = sigmoid.sigmoidGradient(z2);
//            int rowOfsigmoidGradientOfZ2 = sigmoidGradientOfZ2.length;
//            int colOfsigmoidGradientOfZ2 = sigmoidGradientOfZ2[0].length;
//            double[][] d2 = matrix.elementwiseOp(multTheta2FiltransAndD3, sigmoidGradientOfZ2, "*");
//            /*** Step 3 End ***/
//
//            /*** Step 4 start ***/
//            double[][] multD3andtransposeofA2 = matrix.multMatrix(d3, matrix.transpose(a2));
//            double[][] multD2andtransposeofA1 = matrix.multMatrix(d2, matrix.transpose(a1));
//            delta2 = matrix.elementwiseOp(delta2, multD3andtransposeofA2, "+");
//            delta1 = matrix.elementwiseOp(delta1, multD2andtransposeofA1, "+");
            /*** Step 4 end ***/
        }
        Backwardpropopagation backwardpropopagation = new Backwardpropopagation();
        return backwardpropopagation.combineTheta(inputDelta, outputDelta);
    }
}
