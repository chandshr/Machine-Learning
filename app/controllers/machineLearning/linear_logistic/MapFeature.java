package controllers.machineLearning.linear_logistic;

import controllers.machineLearning.all.Matrix;

import java.util.ArrayList;

/**
 * This class is incomplete
 * Created by shrestha on 11/18/2015.
 */
public class MapFeature {

    /**
     * @param A: A with added one
     * @param degree
     * @return: double[][] with added feature e.g if degree = 3: x, y, z, x^2, xy, xy, xz, y^2, yz, z^2
     * this function calls mult() which does the action
     */
    public double[][] mapFeature(double[][] A, int degree){
        int Arow = A.length;
        double[][] result = new double[Arow][];
        Matrix matrix = new Matrix();
        if(degree<1){
            return A;
        }
        else if(degree==1){
//            result = matrix.addColOfOnes(A);
            return result;
        }else{
            for(int i=2; i<=degree; i++){
                for(int j=0; j<A.length; j++){
                    result[j] = mult(A[j]);
                }
            }
            return result;
        }
    }

    /**
     *
     * @param A
     * @return double[][] with added feature e.g if degree = 3: x, y, z, x^2, xy, xy, xz, y^2, yz, z^2
     * and its col are managed as shown in above e.g from left to right; first elements multiplies itself and the element
     * towards its right. It adds col of one to input
     */
    public double[] mult(double[] A){
        double[] result = new double[A.length];
        ArrayList<Double> resultArrl = new ArrayList<Double>();
        /**this adds the input element to resultArrl first and then does mapfeature
         * in next loop. Two loops are used because this helps to insert the first 1 degree element in first col
         */
//        resultArrl.add(1.0);
        for(int i=0; i<A.length; i++){
            resultArrl.add(A[i]);
        }
        for(int i=0; i<A.length; i++){
            for(int j=i; j<A.length; j++){
                resultArrl.add(A[i]*A[j]);
            }
        }

        Matrix matrix = new Matrix();
        result = matrix.convertArrLtoArr(resultArrl);
        return result;
    }
}
