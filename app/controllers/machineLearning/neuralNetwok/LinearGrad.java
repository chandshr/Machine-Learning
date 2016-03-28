package controllers.machineLearning.neuralNetwok;

/**
 * Created by shrestha on 1/7/2016.
 */

/**
 * STEPS: https://theclevermachine.wordpress.com/2014/09/06/derivation-error-backpropagation-gradient-descent-for-neural-networks/
 * Calculated the feed-forward signals from the input to the output.
 * Calculate output error E based on the predictions a_k and the target t_k
 * Backpropagate the error signals by weighting it by the weights in previous layers and the gradients of the associated activation functions
 * Calculating the gradients \frac{\partial E}{\partial \theta} for the parameters based on the backpropagated error signal and the feedforward signals from the inputs.
 * Update the parameters using the calculated gradients \theta \leftarrow \theta - \eta\frac{\partial E}{\partial \theta}
 */
public class LinearGrad {

    private double cost = 0;
    private double[][] J_history;
    private double[][] gradTheta;
    private double[][] pred;

    private int inputlayersize;
    private int hiddenlayersize;
    private int numlabels;

    private double[][] theta1;
    private double[][] theta2;

    public double[][] getGradient(double[][] theta, double[][] delta, int inputlayersize, int hiddenlayersize,
                                  int numlabels, double[][] X, double[][] y, double lambda, double alpha, int iter, boolean isMultiClass){
        this.inputlayersize = inputlayersize;
        this.hiddenlayersize = hiddenlayersize;
        this.numlabels = numlabels;
        LinearNNcost linearNNcost = new LinearNNcost();

        linearNNcost.setMulticlass(isMultiClass);
        int thetaRow = theta.length;
        cost = linearNNcost.cost(theta, inputlayersize, hiddenlayersize,
                numlabels, X, y, lambda);
        for(int p=0; cost>0.1; p++){
            for(int j=0; j<thetaRow; j++){
                theta[j][0] = theta[j][0] - alpha*delta[j][0];
            }
            cost = linearNNcost.cost(theta, inputlayersize, hiddenlayersize,
                    numlabels, X, y, lambda);
            delta = linearNNcost.gettheta();
            System.out.println("Decreasing Cost: "+this.cost);
        }
        System.out.println("Decreasing Cost: "+this.cost);
        pred = linearNNcost.getPred();
        gradTheta = theta;
        setSubTheta(); //theta1 and theta2
        return this.gradTheta;
    }

    public double getCost(){
        return this.cost;
    }

    public double[][] getTheta1() {
        return theta1;
    }

    public double[][] getTheta2() {
        return theta2;
    }

    public void setSubTheta(){
        this.theta1 = new double[hiddenlayersize][inputlayersize+1];
        int thetaCount = 0;
        for(int i=0; i<inputlayersize+1; i++){
            for(int j=0; j<hiddenlayersize; j++){
                theta1[j][i] = gradTheta[thetaCount][0];
                thetaCount++;
            }
        }
        /***** Theta1 end *****/

        /*** Theta2 start ***/
        this.theta2 = new double[numlabels][hiddenlayersize+1];
        for(int i=0; i<hiddenlayersize+1; i++){
            for(int j=0; j<numlabels; j++){
                theta2[j][i] = gradTheta[thetaCount][0];
                thetaCount++;
            }
        }
//        /*** Theta2 end ***/
    }

    public double[][] getPred(){
        return this.pred;
    }
}

