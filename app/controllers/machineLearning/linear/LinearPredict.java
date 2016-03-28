package controllers.machineLearning.linear;

/**
 * Created by shrestha on 11/16/2015.
 */
public class LinearPredict {

    public double predict(double[] input, double[] mean, double[] std, double[][] theta){
        int col = input.length;
        double predict = theta[0][0]; //we need to have X(0)*theta(0) which is 1*theta(0) so, we didn't added one in input[]
        double inputMeanDiff;
        double divByStd;
        for(int i=0; i<mean.length; i++){
            inputMeanDiff = input[i]-mean[i];
            if(std[i]!=0){
                divByStd = inputMeanDiff/std[i];
                predict += divByStd*theta[i+1][0];
            }
            //the below commented line was actual but to avoid NaN when std = 0; above if condition was added
//            divByStd = inputMeanDiff/std[i];
//            predict += divByStd*theta[i+1][0];
        }
        return predict;
    }

    public void getAccuracy(double[][] xTest, double[] mean, double[] std, double[][] theta, double[][] y){
        double[] pred = new double[xTest.length];
        int sum = 0;
        for(int i=0; i<xTest.length; i++){
            pred[i] = predict(xTest[i], mean, std, theta);
            System.out.println("Actual: "+y[i][0]+" predicted:"+pred[i]);
            if((int)pred[i]==(int)y[i][0]){
                sum++;
            }
        }
//        return sum*100/xTest.length;
    }
}
