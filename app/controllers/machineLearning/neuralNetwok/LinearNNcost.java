package controllers.machineLearning.neuralNetwok;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.all.Sigmoid;

/**
 * Created by shrestha on 12/18/2015.
 */
public class LinearNNcost {

    private double[][] theta1;
    private double[][] theta2;
    private double[][] delta;
    private double[][] pred;
    private boolean multiclass; //class greater than equal to 3 //if it has two classes then it is not multiclass

    public double[][] gradtheta;

    public double cost(double[][] theta, int inputlayersize, int hiddenlayersize,
                       int numlabels, double[][] X, double[][] y, double lambda){
        Matrix matrix = new Matrix();
        Sigmoid sigmoid = new Sigmoid();

        int Xrow = X.length;
        int Xcol = X[0].length;

        /***** Theta1 start *****/
        double[][] theta1 = new double[hiddenlayersize][inputlayersize+1];
        int thetaCount = 0;
        for(int i=0; i<inputlayersize+1; i++){
            for(int j=0; j<hiddenlayersize; j++){
                theta1[j][i] = theta[thetaCount][0];
                thetaCount++;
            }
        }
        /***** Theta1 end *****/

        /*** Theta2 start ***/
        double[][] theta2 = new double[numlabels][hiddenlayersize+1];
        for(int i=0; i<hiddenlayersize+1; i++){
            for(int j=0; j<numlabels; j++){
                theta2[j][i] = theta[thetaCount][0];
                thetaCount++;
            }
        }
        /*** Theta2 end ***/

        /****** Part 1 Start *****/
        double[][] a1 = new double[X.length][X[0].length+1];
        for(int i=0; i<X.length; i++){
            a1[i][0] = 1;
            for(int j=0; j<X[0].length; j++){
                a1[i][j+1] = X[i][j];
            }
        }
        double[][] multTheta1a1Trans = matrix.multMatrix(theta1, matrix.transpose(a1));
        double[][] a2 = sigmoid.getSigmoidArr(multTheta1a1Trans);
        a2 = matrix.addOneRowOfOnes(a2);
        double[][] multTheta2A2 = matrix.multMatrix(theta2, a2);
        double[][] h = sigmoid.getSigmoidArr(multTheta2A2);

        int yRow = y.length;

        h = matrix.transpose(h);
        double sumCost = 0;
        for(int i=0; i<y.length; i++){
            sumCost += (y[i][0]-h[i][0])*(y[i][0]-h[i][0]);
        }

        double J = sumCost/yRow;

        /********Regularization Start********/
        double[][] theta1Filtered = new double[theta1.length][theta1[0].length-1];
        double[][] theta2Filtered = new double[theta2.length][theta2[0].length-1];

        double sumSqTheta1 = 0;
        for(int i=0; i<theta1.length; i++){
            for(int j=1; j<theta1[0].length; j++){
                theta1Filtered[i][j-1] = theta1[i][j];
                sumSqTheta1 += (theta1[i][j]*theta1[i][j]);
            }
        }

        double sumSqTheta2 = 0;
        for(int i=0; i<theta2.length; i++){
            for(int j=1; j<theta2[0].length; j++){
                theta2Filtered[i][j-1] = theta2[i][j];
                sumSqTheta2 += (theta2[i][j]*theta2[i][j]);
            }
        }

        double reg = (lambda/(2*yRow))*(sumSqTheta1+sumSqTheta2);
        /********Regularization End********/
        J = J + reg;

        double[][] delta1 = new double[theta1.length][theta1[0].length];
        double[][] delta2 = new double[theta2.length][theta2[0].length];
        double[][] z2 = new double[theta1.length][a1[0].length];
        double[][] a3 = h.clone();
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
            double[][] yt = matrix.transpose(y[i]); //for multiclass
            double[][] d3 = matrix.elementwiseOp(a3, yt, "-");
            /*** step 2 end ***/

            /*** Step 3 Start ***/
            double[][] multTheta2FiltransAndD3 = matrix.multMatrix(matrix.transpose(theta2Filtered), d3);

            double[][] sigmoidGradientOfZ2 = sigmoid.sigmoidGradient(z2);
            int rowOfsigmoidGradientOfZ2 = sigmoidGradientOfZ2.length;
            int colOfsigmoidGradientOfZ2 = sigmoidGradientOfZ2[0].length;
            double[][] d2 = matrix.elementwiseOp(multTheta2FiltransAndD3, sigmoidGradientOfZ2, "*");
            /*** Step 3 End ***/

            /*** Step 4 start ***/
            double[][] multD3andtransposeofA2 = matrix.multMatrix(d3, matrix.transpose(a2));
            double[][] multD2andtransposeofA1 = matrix.multMatrix(d2, matrix.transpose(a1));
            delta2 = matrix.elementwiseOp(delta2, multD3andtransposeofA2, "+");
            delta1 = matrix.elementwiseOp(delta1, multD2andtransposeofA1, "+");
            /*** Step 4 end ***/
        }

        double x = (double) 1/Xrow;
        double[][] theta1Grad = thetaMultByElement(delta1, x);
        double[][] theta2Grad = thetaMultByElement(delta2, x);

        double[][] temp1 = removeFirstColumn(theta1Grad);
        double[][] temp2 = removeFirstColumn(theta2Grad);

        x = (double) lambda/Xrow;
        theta1Filtered = thetaMultByElement(theta1Filtered, x);
        theta2Filtered = thetaMultByElement(theta2Filtered, x);

        temp1 = matrix.elementwiseOp(temp1, theta1Filtered, "+");
        temp2 = matrix.elementwiseOp(temp2, theta2Filtered, "+");

        theta1Grad = matrix.copy(theta1Grad, temp1, 1);
        theta2Grad = matrix.copy(theta2Grad, temp2, 1);

        Backwardpropopagation backwardpropopagation = new Backwardpropopagation();
        this.gradtheta = backwardpropopagation.combineTheta(theta1Grad, theta2Grad);

        delta = backwardpropopagation.combineTheta(delta1, delta2);

        this.theta1 = theta1Grad;
        this.theta2 = theta2Grad;
        pred = h;

        //accuracy start
//        AccuracyFeedforward accuracyFeedforward = new AccuracyFeedforward();
//        accuracyFeedforward.displayAccuracy(a3, X, y);
        //accuracy end
//        System.out.println("NNcost: "+J);
        return J;
        /***** cost positive and negative end ****/
        /****** Part 1 End *****/
    }

    /******divided each element of matrix by second argument*****/
    public double[][] thetaMultByElement(double[][] input, double x){
        int row = input.length;
        int col = input[0].length;
        double[][] result = new double[row][col];

        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                result[i][j] = input[i][j]*x;
            }
        }
        return result;
    }

    /******remove first column*****/
    public double[][] removeFirstColumn(double[][] input){
        int row = input.length;
        int col = input[0].length;
        double[][] result = new double[row][col-1];

        for(int i=0; i<row; i++){
            for(int j=1; j<col; j++){
                result[i][j-1] = input[i][j];
            }
        }
        return result;
    }

//    public double[][] getTheta2() {
//        return theta2;
//    }
//
//    public void setTheta2(double[][] theta2) {
//        this.theta2 = theta2;
//    }
//
//    public double[][] getTheta1() {
//        return theta1;
//    }
//
//    public void setTheta1(double[][] theta1) {
//        this.theta1 = theta1;
//    }

    public double[][] predict(double[][] X, double[][] theta1, double[][] theta2){
        int Xrow = X.length;
        Matrix matrix = new Matrix();
        Sigmoid sigmoid = new Sigmoid();
        double[][] a1;
        double[][] a2;
        double[][] z2;
        double[][] z3;
        double[][] a3;
        double[][] d3;
        double[][] pred = new double[Xrow][1];
        for(int i=0; i<Xrow; i++){
            double[] b = X[i];
            double[][] a = matrix.transpose(b);
            a1 = matrix.addOneRowOfOnes(a);

            z2 = matrix.multMatrix(theta1, a1);
            a2 = matrix.addOneRowOfOnes(sigmoid.getSigmoidArr(z2));

            z3 = matrix.multMatrix(theta2, a2);
            a3 = sigmoid.getSigmoidArr(z3);
            double max = a3[0][0];
            if(a3[1][0]>max){
                pred[i][0] = 2;
            }else{
                pred[i][0] = 1;
            }
        }
        return pred;
    }

    public double[][] getPred(){
        return this.pred;
    }

    public void setMulticlass(boolean isMulticlass){
        multiclass = isMulticlass;
    }

    public boolean getMulticlass(){
        return multiclass;
    }

    public double[][] getDelta(){
        return delta;
    }

    public double[][] getTheta1(){
        return theta1;
    }

    public double[][] getTheta2(){
        return theta2;
    }

    public double[][] gettheta() {
        return gradtheta;
    }
}
