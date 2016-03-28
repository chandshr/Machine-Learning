package controllers.machineLearning.neuralNetwok;

import controllers.machineLearning.all.Matrix;

/**
 * Created by shrestha on 12/22/2015.
 */
public class ComputeNumericalGradient {

    public double[][] compute(double[][] theta,int inputlayersize, int hiddenlayersize,
                        int numlabels, double[][] X, double[][] y, double lambda, boolean isMultiClass){
        double[][] numGrad = new double[theta.length][1];
        double[][] perturb = new double[theta.length][1];
        double e = 1e-4;
        int row = theta.length;
        int col = theta[0].length;
        Matrix matrix = new Matrix();
        NNCostFunction NNCostFunction = new NNCostFunction();
        NNCostFunction.setMulticlass(isMultiClass);
        for(int i=0; i<row; i++){
            perturb[i][0] = e;
            double[][] diffTheta = matrix.elementwiseOp(theta, perturb, "-");
            double[][] positiveTheta = matrix.elementwiseOp(theta, perturb, "+");

            double loss1 = NNCostFunction.cost(diffTheta, inputlayersize, hiddenlayersize,
            numlabels, X, y, lambda);
            double loss2 = NNCostFunction.cost(positiveTheta, inputlayersize, hiddenlayersize,
                    numlabels, X, y, lambda);
            numGrad[i][0] = (loss2-loss1)/(2*e);
//            System.out.println("numGrad: "+i+", "+numGrad[i][0]);
            perturb[i][0] = 0;
        }
        return numGrad;
    }
}
