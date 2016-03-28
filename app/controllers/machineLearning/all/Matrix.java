package controllers.machineLearning.all;

import java.util.ArrayList;

/**
 * Created by shrestha on 12/14/2015.
 */
public class Matrix {

    public double[][] transpose(double[][] input){
        int row = input.length;
        int col = input[0].length;
        double[][] trans = new double[col][row];
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                trans[j][i] = input[i][j];
            }
        }
        return trans;
    }

    public double[][] transpose(double[] input){
        double[][] transpose = new double[input.length][1];
        for(int i=0; i<input.length; i++){
            transpose[i][0] = input[i];
        }
        return transpose;
    }

    public double[][] addColOfOnes(double[][] input){
        int inputRow = input.length;
        int inputCol = input[0].length;
        double[][] result = new double[inputRow][inputCol+1];

        for(int i=0; i<inputRow; i++){
            result[i][0] = 1;
        }
        result = copy(result, input, 1);
        return result;
    }

    public double[][] addOneRowOfOnes(double[][] input){
        int inputRow = input.length;
        int inputCol = input[0].length;
        double[][] result = new double[inputRow+1][inputCol];
        for(int j=0; j<input[0].length; j++){
            result[0][j] = 1;
        }
        for(int i=0; i<inputRow; i++){
            for(int j=0; j<inputCol; j++){
                result[i+1][j] = input[i][j];
            }
        }
        return result;
    }

    public double[][] multMatrix(double[][] firstMatrix, double[][] secMatrix){
        int firstRow = firstMatrix.length;
        int firstCol = firstMatrix[0].length;
        int secRow = secMatrix.length;
        int secCol = secMatrix[0].length;

        if(firstCol != secRow){
            System.out.println("no. of column of first matrix = "+firstCol+" not equal to no. of row of second matrix = "+secRow);
            return null;
        }
        double[][] result = new double[firstRow][secCol];
        for(int k=0; k<secCol; k++){
            for(int i=0; i<firstRow; i++){
                double sum = 0;
                for(int j=0; j<firstCol; j++){
                    sum += firstMatrix[i][j]*secMatrix[j][k];
                }
                result[i][k] = sum;
            }
        }
        return result;
    }

    public double[][] getLogOfMatrix(double[][] input){
        int row = input.length;
        int col = input[0].length;
        double[][] logOfGivenArr = new double[row][col];
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                logOfGivenArr[i][j] = Math.log(input[i][j]);
            }
        }
        return logOfGivenArr;
    }

//    public double[][] difference(double[][] input1, double[][] input2){
//        double[][] diff = new double[input1.length][input1[0].length];
//        for(int i=0; i<input1.length; i++){
//            for(int j=0; j<input1[0].length; j++){
//                diff[i][j] = input1[i][j] - input2[i][j];
//            }
//        }
//        return diff;
//    }
//
//    public double[][] add(double[][] input1, double[][] input2) {
//        double[][] add = new double[input1.length][input1[0].length];
//        for (int i = 0; i < input1.length; i++) {
//            for (int j = 0; j < input1[0].length; j++) {
//                add[i][j] = input1[i][j] + input2[i][j];
//            }
//        }
//        return add;
//    }

    /****** multiply two matrices element wise *****/
    public double[][] elementwiseOp(double[][] inputA, double[][] inputB, String op){
        int row = inputA.length;
        int col = inputA[0].length;
        double[][] result = new double[row][col];
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                if(op=="+")
                    result[i][j] = inputA[i][j] + inputB[i][j];
                else if(op=="-")
                    result[i][j] = inputA[i][j] - inputB[i][j];
                else if(op=="*")
                    result[i][j] = inputA[i][j] * inputB[i][j];
            }
        }
        return result;
    }

    /***copy content Start
     * copies the content in result[][] from input[][]; starts copying content from argument position
     * ***/
    public double[][] copy(double[][] result, double[][] input, int position){
        int row = input.length;
        int col = input[0].length;
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                result[i][position+j] = input[i][j];
            }
        }
        return result;
    }
    /***copy content End***/

    /*****copies element values row by row START
     * thetaFiltered = [0; theta(2:end)];
     * *****/
    public double[][] copyFromRowPosition(double[][] result, double[][] input, int position){
        int row = input.length;
        int col = input[0].length;
        for(int i=position; i<row; i++){
            result[i][0] = input[i][0];
        }
        return result;
    }
    /*****copies element values row by row END *****/

    /**
     * normalize the given matrix
     * n = norm(v,p) returns the vector norm defined by sum(abs(v)^p)^(1/p), where p is any positive
     * here p is 2
     */

    public double getNorm(double[][] input){
        int row = input.length;
        int col = input[0].length;

        double squareSum = 0;
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                squareSum += input[i][j]*input[i][j];
            }
        }
        return Math.sqrt(squareSum);
    }

    public double[][] matrixDivideorMultBy(double[][] input, double x, String operator){
        int row = input.length;
        int col = input[0].length;
        double[][] result = new double[row][col];

        if(operator=="/"){
            for(int i=0; i<row; i++){
                for(int j=0; j<col; j++){
                    result[i][j] = (double) input[i][j]/x;
                }
            }
        }else if(operator=="*"){
            for(int i=0; i<row; i++){
                for(int j=0; j<col; j++){
                    result[i][j] = (double) input[i][j]*x;
                }
            }
        }

        return result;
    }

    /**  get array of column(features) from a given two dimensional array **/
    public double[] getColArr(double[][] input, int colNo){
        int row = input.length;
        double[] result = new double[row];
        for(int i=0; i<row; i++){
            result[i] = input[i][colNo];
        }
        return result;
    }

    public double oneDimensionalOp(double[] input, String op){
        int row = input.length;
        double max = input[0];
        double min = input[0];
        if(op=="max"){
            for(int i=1; i<row; i++){
                if(input[i]>max){
                    max = input[i];
                }
            }
            return max;
        }else if(op=="min"){
            for(int i=1; i<row; i++){
                if(input[i]<min){
                    min = input[i];
                }
            }
            return min;
        }
        return 0;
    }

    /**
     * convert arraylist to array
     */
    public double[] convertArrLtoArr(ArrayList<Double> input){
        int row = input.size();
        double[] result = new double[row];
            for(int i=0; i<row; i++){
                result[i] = input.get(i);
        }
        return result;
    }

    /**
     *
     * @param y : it contains value of certain range like {1, 2, 3, 4}
     * @param value : value we are looking for in y
     * @return return binary class with y[][] = 1; if it equals to the value we are looking for else y[][] = 0
     */
    public double[][] binaryClassfromMultiClass(double[][] y, int value){
        double[][] result = new double[y.length][1];
        for(int i=0; i<y.length; i++){
            if(y[i][0]!=value){
                result[i][0] = 0;
            }else{
                result[i][0] = y[i][0];
            }
        }
        return result;
    }

    /**
     * LOGISTIC AND BACKPROPAGATION
     * if y==4; y=0
     * if y==3; y=1
     * change the value of arr to make value of y = 0 or 1 if it holds only two class e.g. to represent benign and malignant
     * change its value to y = 1, 2, 3, if it holds more than two class which are not in proper order because in multiclass
     * problem the 1th position is 1 and others are zero and 2th position is one and others are zero
     * */
    public double[][] changeArrVal(double[][]y, int[] x){
        if(x.length==2){
            for(int i=0; i<y.length; i++){
                if(y[i][0]==x[0]){
                    y[i][0] = 0;
                }else{
                    y[i][0] = 1;
                }
            }
        }else if(x.length>2){
            for(int i=0; i<y.length; i++){
                for(int j=0; j<x.length; j++){
                    if(y[i][0]==x[j]){
                        y[i][0] = j+1;
                    }
                }
            }
        }
        return y;
    }

    public int[] strToIntArr(String input){
        input = input.replaceAll("\\s+","");
        String[] inputArrStr = input.split(",");
        int[] inputArrInt = new int[inputArrStr.length];
        for(int i=0; i<inputArrStr.length; i++){
            inputArrInt[i] = Integer.parseInt(inputArrStr[i]);
        }
        return inputArrInt;
    }

    public int predictClass(int[] classArr, int predClass){
        if(classArr.length==2){
            return classArr[predClass];
        }else if(classArr.length>2){
            return classArr[predClass-1];
        }
        return -1;
    }
}
