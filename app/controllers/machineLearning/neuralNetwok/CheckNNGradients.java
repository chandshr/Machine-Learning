package controllers.machineLearning.neuralNetwok;

import controllers.machineLearning.all.Matrix;

/**
 * Created by shrestha on 12/22/2015.
 */
public class CheckNNGradients {

    public void checkNeuralNetwork(double lamda, int input_layer_size, int hidden_layer_size, int num_labels, double[][] X, double[][] y, double[][] theta1, double[][] theta2, double[][] theta, boolean isMulticlass){
//    public void checkNeuralNetwork(double lamda){
//        int input_layer_size = 3;
//        int hidden_layer_size = 5;
//        int num_labels = 3;
//        int m = 5;

        DebugInitializeWeights debugInitializeWeights = new DebugInitializeWeights();

        //generate random test data
//        double[][] theta1 = debugInitializeWeights.createMatrix(hidden_layer_size, input_layer_size);
//        double[][] theta2 = debugInitializeWeights.createMatrix(num_labels, hidden_layer_size);
//        double[][] X = debugInitializeWeights.createMatrix(m, input_layer_size-1);
//        double[][] X = {{2, 22, 34}, {3, 22, 34}, {1, 22, 34}, {2, 22, 34}, {3, 22, 34}}; //identical to octave y variable
//        double[][] y = {{2}, {3}, {1}, {2}, {3}}; //identical to octave y variable

//        Backwardpropopagation backwardpropopagation = new Backwardpropopagation();
//        double[][] theta = backwardpropopagation.combineTheta(theta1, theta2);
        NNCostFunction NNCostFunction = new NNCostFunction();
        NNCostFunction.setMulticlass(isMulticlass);

        double cost = NNCostFunction.cost(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda);
        double[][] grad = NNCostFunction.gettheta();
//        System.out.println("Gradcost check: "+cost);

//        System.out.println("test1: "+grad.length+"  test2: "+theta.length);

//        System.out.println(theta[theta.length-1][0]);
        ComputeNumericalGradient computeNumericalGradient = new ComputeNumericalGradient();
        double[][] numgrad = computeNumericalGradient.compute(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, isMulticlass);

        Matrix matrix = new Matrix();
        double[][] sumofNumgradAndGrad = matrix.elementwiseOp(numgrad, grad, "+");
        double[][] diffofNumgradAndGrad = matrix.elementwiseOp(numgrad, grad, "-");

        double diff = (double) matrix.getNorm(diffofNumgradAndGrad)/matrix.getNorm(sumofNumgradAndGrad);
        System.out.println("CheckNNGradient.java if backpropagation is correct diff value should be less than 1e-9:   "+diff);
    }
}
