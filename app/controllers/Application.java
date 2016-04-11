package controllers;

import controllers.machineLearning.all.Matrix;
import controllers.machineLearning.all.PlotDecisionBoundary;
import controllers.machineLearning.all.TextToArr;
import controllers.machineLearning.learningCurve.LearningCurve;
import controllers.machineLearning.linear.*;
import controllers.machineLearning.linear_logistic.FeatureNormalize;
import controllers.machineLearning.linear_logistic.RegularizedGradient;
import controllers.machineLearning.logistic.LogisticCost;
import controllers.machineLearning.logistic.LogisticGradientDescent;
import controllers.machineLearning.logistic.LogisticPredict;
import controllers.machineLearning.logistic.LogisticRegCost;
import controllers.machineLearning.neuralNetwok.*;
import org.apache.http.protocol.HTTP;
import play.*;
import play.data.DynamicForm;
import play.data.Form;
import play.mvc.*;
import play.mvc.Http.*;


import scala.util.parsing.json.JSONObject;
import views.html.*;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.atomic.DoubleAccumulator;

public class Application extends Controller {

    public static String accuracy;
    public static String predict;
    public static String cost;

    private static String uploadDir = Play.application().configuration().getString("myUploadDir");

    public static Result index() {
        return ok(home.render());
    }

    public static Result algorithm() {
        return ok(algorithm.render());
    }

    public static Result linearRegression(){
        return ok(linearRegression.render());
    }

    public static String header(){
        String header = "<html>\n" +
                "    <head>\n" +
                "        <title>@title</title>\n" +
                "        <link rel=\"stylesheet\" media=\"screen\" href=\"@routes.Assets.at(\"stylesheets/bootstrap.css\")\">\n" +
                "        <link rel=\"shortcut icon\" type=\"image/png\" href=\"@routes.Assets.at(\"images/favicon.png\")\">\n" +
                "        <link href=\"//maxcdn.bootstrapcdn.com/bootstrap/3.3.0/css/bootstrap.min.css\" rel=\"stylesheet\">\n" +
                "        <link href=\"//maxcdn.bootstrapcdn.com/font-awesome/4.2.0/css/font-awesome.min.css\" rel=\"stylesheet\">\n" +
                "        <script src=\"@routes.Assets.at(\"javascripts/jquery-1.9.0.min.js\")\" type=\"text/javascript\"></script>\n" +
                "    </head>\n" +
                "    <head>\n" +
                "    <meta charset=\"utf-8\">\n" +
                "    <title>@title</title>\n" +
                "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n" +
                "    <meta name=\"description\" content=\"\">\n" +
                "    <meta name=\"author\" content=\"\">\n" +
                "\n" +
                "    <!-- Le styles -->\n" +
                "    <link rel=\"stylesheet\" media=\"screen\" href=\"@routes.Assets.at(\"stylesheets/bootstrap.css\")\">\n" +
                "    <link rel=\"stylesheet\" media=\"screen\" href=\"@routes.Assets.at(\"stylesheets/bootstrap-responsive.css\")\">\n" +
                "    <link rel=\"stylesheet\" media=\"screen\" href=\"@routes.Assets.at(\"stylesheets/main.css\")\">\n" +
                "    </head>\n" +
                "    <body>\n" +
                "        <div class=\"container-narrow\">";
        return header;
    }

    public static String footer(){
        String footer = "<div class=\"footer\">\n" +
                "                <div class=\"social\">\n" +
                "                    <ul>\n" +
                "                        <li><a href=\"chandani\"><i class=\"fa fa-lg fa-facebook\"></i></a></li>\n" +
                "                        <li><a href=\"chandani\"><i class=\"fa fa-lg fa-twitter\"></i></a></li>\n" +
                "                        <li><a href=\"chandani\"><i class=\"fa fa-lg fa-github\"></i></a></li>\n" +
                "                        <li><a href=\"chandani\"><i class=\"fa fa-lg fa-linkedin\"></i></a></li>\n" +
                "                    </ul>\n" +
                "                </div>\n" +
                "                <hr>\n" +
                "                <span class=\"copyright\">Copyright &copy; CHANDANI 2016</span>\n" +
                "            </div>\n" +
                "        </div>\n" +
                "    </body>\n" +
                "</html>";
        return footer;
    }

    public static Result outputLinearRegression() {
        Http.MultipartFormData body = request().body().asMultipartFormData();
        Http.MultipartFormData.FilePart datafile = body.getFile("dataFile");

        String trainFilename;
        if (datafile != null) {
            String fileName = datafile.getFilename();
            String contentType = datafile.getContentType();
            File file = datafile.getFile();
            file.renameTo(new File(uploadDir, fileName));
            trainFilename = uploadDir+"/"+fileName;
        } else {
            flash("error", "Missing file");
            return ok("Please upload file home page");
        }

        DynamicForm df = play.data.Form.form().bindFromRequest();
        String featureVal = df.get("featureArr");
        featureVal = featureVal.replaceAll("\\s+","");
        String[] featureArr = featureVal.split(",");
        String outputCol = df.get("outputCol");
        int outputColNo = Integer.parseInt(outputCol);
        String etaStr = df.get("eta");
        String iterStr = df.get("iter");
        double alpha = Double.valueOf(etaStr);
        int iter = Integer.valueOf(iterStr);


        response().setContentType("text/html");
        String output = new String();
        output = output.concat("<h2>Linear Regression</h2><table style='width:40%'>");
        String filepath = trainFilename;
        String jsonfilePath = uploadDir+"/linearInput.json";

        /***** 1. Learning Curve Start ******/
        String seperator = ",";
        LearningCurve learningCurve = new LearningCurve();
        learningCurve.initializeTrainTestcrossValidData(filepath, jsonfilePath, seperator, outputColNo);//initialize feature normalized train, test and cross validation data
        double[][] xTrain =learningCurve.getxTrain();
        double[][] yTrain = learningCurve.getyTrain();
        double[][] xCross =learningCurve.getxCross();
        double[][] yCross = learningCurve.getyCross();
        double[][] xTest = learningCurve.getxTest();
        double[][] yTest = learningCurve.getyTest();

        FeatureNormalize featureNormalize = new FeatureNormalize();
        Matrix matrix = new Matrix();

        xTrain = featureNormalize.getFeatureNormalize(xTrain);
        double[] mean = featureNormalize.getMean();
        double[] std = featureNormalize.getStd();

        xTest = featureNormalize.getFeatureNormalize(xTest);
        xCross = featureNormalize.getFeatureNormalize(xCross);

        xTrain = matrix.addColOfOnes(xTrain);
        /**
         * don't add ones in xtest and xcross because it is already added in predict function
         */
        /****** Learning Curve End *****/

        /********** 2. Gradient Descent start***************/
        LinearGradientDescent gradientDescent = new LinearGradientDescent();

//        double lambda = 0.035;


        double[][] initial_theta = new double[xTrain[0].length][1]; //no. of elements in theta is size of no. of col of X + 1; first is 1
        double[][] theta = gradientDescent.getGradient(initial_theta, xTrain, yTrain, alpha, iter);
        LinearCost linearCost = new LinearCost();
        double cost = gradientDescent.getCost();
//        System.out.println("Linear RegressionInterface cost from "+xTrain.length+" train instances at gradient descent: "+cost);
        output = output.concat("<tr><td>Training Error</td><td>"+cost+"</td></tr>");
        Application.cost = Double.toString(cost);
//        System.out.println("Gradient Descent value stored in theta[][]");
        /********** Gradient Descent end ***************/

        /**********Prediction Start ************/
        LinearPredict linearPredict = new LinearPredict();
        double[] testInput = xTest[0];
        double[] postX = new double[featureArr.length];
        for(int i=0; i<featureArr.length; i++){
            postX[i] = Double.parseDouble(featureArr[i]);
        }
        /**
         * trainFeatureNormalize deals with xTrain data. So, it gives mean and std related to training data
         * It gives maximum accuracy. So, use train feature
         */

//        System.out.println("Linear Regression predicted value "+linearPredict.predict(testInput, mean, std, theta)+" for actual = "+yTest[0][0]);
//        output = output.concat("{\"predicted\":\""+linearPredict.predict(testInput, mean, std, theta)+"\"}");

        double predict = linearPredict.predict(postX, mean, std, theta);
        output = output.concat("<tr><td>Predicted Value</td><td>"+predict+"</td></tr>");
        Application.predict = Double.toString(predict);
//        output = output.concat("{\"actual\":\""+yTest[0][0]+"\"}");
        /**********Prediction End ************/

        //predict start
        linearPredict.getAccuracy(xTest, mean, std, theta, yTest);
        //predict end

        /********** Linear Regularized Start**********/
        LinearRegCost linearRegCost = new LinearRegCost();
        double lamda = .1;
        double regCost = linearRegCost.regCost(xTrain, yTrain, theta, lamda);
//        System.out.println("Linear Regularized cost: "+regCost);

        RegularizedGradient regularizedGradient = new RegularizedGradient();
        double[][] linearGrad = regularizedGradient.getGradient(theta, xTrain, yTrain, lamda, "machineLearning/linear");
        double regularizedGradientCost = linearRegCost.regCost(xTrain, yTrain, linearGrad, lamda);
//        System.out.println("Linear Regualarized cost after implementing regularized gradient "+regularizedGradientCost);
//        output = output.concat("<tr><td>Regularized Gradient Cost</td><td>"+regularizedGradientCost+"</td></tr>");
        /********** Linear Regularized End**********/

        /******Learning Curve Start*******/
//        HashMap<Double, Double> traincostLambdaLinearHash = learningCurve.getCostLamdaHash(xTrain, yTrain, theta, 0.001, 2, linearCost); //lambda = 0.32 optimum //lambda = 0.64 (cost=nan)
//
//        Matrix matrix = new Matrix();
//        xCross = matrix.addColOfOnes(xCross);//add column of ones in crossvalidation
//
//        HashMap<Double, Double> crossValidcostLambdaLinearHash = learningCurve.getCostLamdaHash(xCross, yCross, theta, 0.001, 2, linearCost); //lambda = 0.32 optimum //lambda = 0.64 (cost=nan)
//
//        System.out.println("The result of train and crossvalidation was optimum so, iter = 400 and lambda = 0.1 is the optimum value for this parkinson data set");
//
//        HashMap<Integer, Double> traincostDegreeLinearHash = learningCurve.getCostDegreeHash(xTrain, yTrain, theta,  1, linearCost);
//        HashMap<Integer, Double> crossValidcostDegreeLinearHash = learningCurve.getCostDegreeHash(xCross, yCross, theta,  1, linearCost);
//
//        System.out.println("With Degree: 2 train and cross validation hashcost learning curve was same. But, as the degree was raised to 8 'OutOfMemoryError' occured");
//
//        HashMap<Integer, Double> traincostTraincountLinearHash = learningCurve.getCostTraincountHash(xTrain, yTrain, theta, LearningCurve.getCrossValidRow(), linearCost);
//        HashMap<Integer, Double> crossValidcostTraincountLinearHash = learningCurve.getCostTraincountHash(xCross, yCross, theta, LearningCurve.getCrossValidRow(), linearCost);
//
//        System.out.println("The learning curve drawn with training count verses cost shows optimum value on hashmap");
//
//        System.out.println("\n*******Learning Curve Result: No Bias and Variance. Optimum value at lambda = 0.1, iter = 400, degree = 1, training count is optimum ******");
        /******Learning Curve End******/
        output = output.concat("</table>");
        return ok(linearResult.render());
    }

    public static Result setBiLogistic(){
        return ok(uniLogistic.render());
    }

    public static Result setMultiLogistic(){
        return ok(multiLogistic.render());
    }

    public static Result biLogisticRegression(){
        Http.MultipartFormData body = request().body().asMultipartFormData();
        Http.MultipartFormData.FilePart datafile = body.getFile("dataFile");

        String trainFilename;
        if (datafile != null) {
            String fileName = datafile.getFilename();
            String contentType = datafile.getContentType();
            File file = datafile.getFile();
            file.renameTo(new File(uploadDir, fileName));
            trainFilename = uploadDir+"/"+fileName;
        } else {
            flash("error", "Missing file");
            return ok("Please upload file home page");
        }

        DynamicForm df = play.data.Form.form().bindFromRequest();
        String outputCol = df.get("outputCol");
        int outputColNo = Integer.parseInt(outputCol);

        String featureVal = df.get("featureArr");
        String etaStr = df.get("eta");
        String iterStr = df.get("iter");
        String posVal = df.get("pos");
        String negVal = df.get("neg");

        int[] valArr = new int[2];
        valArr[0] = Integer.parseInt(negVal);
        valArr[1] = Integer.parseInt(posVal);


        double alpha = Double.valueOf(etaStr);
        int iter = Integer.valueOf(iterStr);

        featureVal = featureVal.replaceAll("\\s+","");
        String[] featureArr = featureVal.split(",");

        double[][] postX = new double[1][featureArr.length];
        for(int i=0; i<featureArr.length; i++){
            postX[0][i] = Double.parseDouble(featureArr[i]);
        }

        Matrix matrix = new Matrix();
        FeatureNormalize featureNormalize = new FeatureNormalize();
//        postX = featureNormalize.getFeatureNormalize(postX);
//        postX = matrix.addColOfOnes(postX);
        System.out.println("chan "+postX[0].length);

        response().setContentType("text/html");
        String output = new String();
        output = output.concat("<h2>Logistic Regression</h2><table style='width:40%'>");
        String filepath = trainFilename;
        String jsonfilePath = uploadDir+"/linearInput.json";

        LearningCurve learningCurve = new LearningCurve();
        String seperator = ",";
        learningCurve.initializeTrainTestcrossValidData(filepath, jsonfilePath, seperator, outputColNo);//initialize feature normalized train, test and cross validation data
        double[][] xTrain =learningCurve.getxTrain();
        double[][] yTrain = learningCurve.getyTrain();
        double[][] xTest = learningCurve.getxTest();
        double[][] yTest = learningCurve.getyTest();
        double[][] xCross = learningCurve.getxCross();
        double[][] yCross = learningCurve.getyCross();

        yTrain = matrix.changeArrVal(yTrain, valArr);
        yTest = matrix.changeArrVal(yTest, valArr);
        yCross = matrix.changeArrVal(yCross, valArr);
        System.out.println("xtest "+xTest[0].length);
        xTest = featureNormalize.getFeatureNormalize(xTest);
        xCross = featureNormalize.getFeatureNormalize(xCross);

        xTrain = matrix.addColOfOnes(xTrain);
        xTest = matrix.addColOfOnes(xTest);
        xCross = matrix.addColOfOnes(xCross);

        final double[][] initalTheta = new double[xTrain[0].length][1];
//        double alpha = 0.1; //for bike data get the cost for classes nearly same and then we will have high accuracy
        LogisticCost logisticCost = new LogisticCost();
        double initialCost = logisticCost.getCost(xTrain, yTrain, initalTheta);
        double[][] initialGrad = logisticCost.getGrad();
        /*** Initial Cost End ***/

        /***********iterations to find the optimum theta(gradient Descent) start**********/
//        int iter = 400; //60% accuracy and alpha 0.001
        LogisticGradientDescent gradientDescent = new LogisticGradientDescent();
        double[][] gradTheta;
        LogisticPredict logisticPredict = new LogisticPredict();
        int[][] pred;

        gradTheta = gradientDescent.getGradientDescent(xTrain, yTrain, iter, alpha);
        double gradCost = gradientDescent.getCost();
        System.out.println(xTrain.length+" train instances Cost: "+gradCost);
        /***********Logistic Predictions Start**********/
        int xTestRow = xTest.length;
        int xTestCol = xTest[0].length;
        double finalCost = logisticCost.getCost(xTrain, yTrain, gradTheta);
        output = output.concat("<tr><td>Error</td><td>"+finalCost+"</td></tr>");

        System.out.println("xtest len "+xTest[0].length);
        pred = logisticPredict.predict(gradTheta, xTest);
        postX = matrix.addColOfOnes(postX);
        int[][] predResult = logisticPredict.predict(gradTheta, postX);

        System.out.println(predResult[0][0]);

        int outputVal = Integer.parseInt(negVal);
        if(predResult[0][0]==1){
            outputVal =  Integer.parseInt(posVal);
        }
        output = output.concat("<tr><td>Predicted Class</td><td>"+outputVal+"</td></tr>");
        double accPercent = logisticPredict.accuracy(pred, yTest);
        output = output.concat("<tr><td>Training Accuracy</td><td>"+accPercent+"%</td></tr>");
        output = output.concat("</table>");
        return ok(output);
    }

    public static Result multiLogisticRegression(){
        Http.MultipartFormData body = request().body().asMultipartFormData();
        Http.MultipartFormData.FilePart datafile = body.getFile("dataFile");

        String trainFilename;
        if (datafile != null) {
            String fileName = datafile.getFilename();
            String contentType = datafile.getContentType();
            File file = datafile.getFile();
            file.renameTo(new File(uploadDir, fileName));
            trainFilename = uploadDir+"/"+fileName;
        } else {
            flash("error", "Missing file");
            return ok("Please upload file home page");
        }

        DynamicForm df = play.data.Form.form().bindFromRequest();

        String featureVal = df.get("featureArr");
        String etaStr = df.get("eta");
        String iterStr = df.get("iter");
        String outputCol = df.get("outputCol");
        int outputColNo = Integer.parseInt(outputCol);


        double alpha = Double.valueOf(etaStr);
        int iter = Integer.valueOf(iterStr);

        featureVal = featureVal.replaceAll("\\s+","");
        String[] featureArr = featureVal.split(",");
        String classVal = df.get("logisticClassArr");

        Matrix matrix = new Matrix();
        int[] classArr = matrix.strToIntArr(classVal);

        double[][] postX = new double[1][featureArr.length];
        for(int i=0; i<featureArr.length; i++){
            postX[0][i] = Double.parseDouble(featureArr[i]);
        }

        response().setContentType("text/html");
        String output = new String();
        output = output.concat("<h2>Logistic Regression</h2><table style='width:40%'>");
        String filepath = trainFilename;
        String jsonfilePath = uploadDir+"/linearInput.json";

        LearningCurve learningCurve = new LearningCurve();
        String seperator = ",";
        learningCurve.initializeTrainTestcrossValidData(filepath, jsonfilePath, seperator, outputColNo);//initialize feature normalized train, test and cross validation data

        double[][] xTrain =learningCurve.getxTrain();
        double[][] xTest = learningCurve.getxTest();
        double[][] xCross = learningCurve.getxCross();

        double[][] yTrain = learningCurve.getyTrain();
        double[][] yTest = learningCurve.getyTest();
        double[][] yCross = learningCurve.getyCross();

        yTrain = matrix.changeArrVal(yTrain, classArr);
        yTest = matrix.changeArrVal(yTest, classArr);
        yCross = matrix.changeArrVal(yCross, classArr);


        FeatureNormalize featureNormalize = new FeatureNormalize();

        xTrain = featureNormalize.getFeatureNormalize(xTrain);
        postX = featureNormalize.predFeatureNormalize(postX, featureNormalize.getStd(), featureNormalize.getMean());
        xTest = featureNormalize.getFeatureNormalize(xTest);
        xCross = featureNormalize.getFeatureNormalize(xCross);

        xTrain = matrix.addColOfOnes(xTrain);
        xTest = matrix.addColOfOnes(xTest);
        xCross = matrix.addColOfOnes(xCross);
        postX = matrix.addColOfOnes(postX);

//        yTrain = matrix.changeArrVal(yTrain, valArr);
//        yTest = matrix.changeArrVal(yTest, valArr);
//        yCross = matrix.changeArrVal(yCross, valArr);
//        if(featureArr.length+1!=xTrain[0].length){
//            return ok("number of features not equal to training data features"+featureArr.length+" "+xTrain[0].length);
//        }

        final double[][] initalTheta = new double[xTrain[0].length][1];
//        double alpha = 0.1; //for bike data get the cost for classes nearly same and then we will have high accuracy
        LogisticCost logisticCost = new LogisticCost();
        double initialCost = logisticCost.getCost(xTrain, yTrain, initalTheta);
        double[][] initialGrad = logisticCost.getGrad();
        /*** Initial Cost End ***/

        /***********iterations to find the optimum theta(gradient Descent) start**********/
//        int iter = 400; //60% accuracy and alpha 0.001
        LogisticGradientDescent gradientDescent = new LogisticGradientDescent();
        double[][] gradTheta;
        LogisticPredict logisticPredict = new LogisticPredict();
        int[][] pred;

        pred = new int[yTest.length][1];
        ArrayList<double[][]> grad = new ArrayList();
        ArrayList<double[][]> predArrL = new ArrayList();
        ArrayList<Double> resultPred = new ArrayList();

        if(classArr.length>2){
            for(int j=0; j<classArr.length; j++){
                int yClass = Integer.valueOf(classArr[j]);
                double[][] yVal = matrix.binaryClassfromMultiClass(yTrain, yClass);
                gradTheta = gradientDescent.getGradientDescent(xTrain, yVal, iter, alpha);
                grad.add(gradTheta);
                double finalCost = logisticCost.getCost(xTrain, yTrain, gradTheta);
                output = output.concat("<tr><td>Error in Class "+classArr[j]+"</td><td>"+finalCost+"</td></tr>");
                if(j ==0){
                    Application.cost = "class "+classArr[j]+" = "+finalCost+"\n";
                }else{
                    Application.cost = Application.cost+"class "+classArr[j]+" = "+finalCost+"\n";
                }
                double[][] predClass = logisticPredict.predictMultiClass(gradTheta, xTest, yClass);
                predArrL.add(predClass);
                resultPred.add(logisticPredict.predictMultiClass(gradTheta, postX, yClass)[0][0]);
            }
            for (int i = 0; i < yTest.length; i++) {
                ArrayList<Double> maxArr = new ArrayList<Double>();
                double max = predArrL.get(0)[i][0];
                pred[i][0] = 1;
                for(int j=1; j<classArr.length; j++){
                    if(predArrL.get(j)[i][0]>max){
                        max = predArrL.get(j)[i][0];
                        pred[i][0] = j+1;
                    }
                }
            }
            double max = resultPred.get(0);
            int resultClass = 1;
            for(int i=1; i<resultPred.size(); i++){
                if(resultPred.get(i)>max){
                    max = resultPred.get(i);
                    resultClass = i+1;
                }
            }
            output = output.concat("<tr><td>Predicted Class</td><td>"+resultClass+"</td></tr>");
            Application.predict = Integer.toString(resultClass);

        }else{
            gradTheta = gradientDescent.getGradientDescent(xTrain, yTrain, iter, alpha);
            double gradCost = gradientDescent.getCost();
            System.out.println(xTrain.length+" train instances Cost: "+gradCost);
            Application.cost = Double.toString(gradCost);
            /***********Logistic Predictions Start**********/
            int xTestRow = xTest.length;
            int xTestCol = xTest[0].length;
            double finalCost = logisticCost.getCost(xTrain, yTrain, gradTheta);
            output = output.concat("<tr><td>Error</td><td>"+finalCost+"</td></tr>");

            pred = logisticPredict.predict(gradTheta, xTest);

            int[][] predResult = logisticPredict.predict(gradTheta, postX);

            int predClass = predResult[0][0];
            predClass = matrix.predictClass(classArr, predClass);
            output = output.concat("<tr><td>Predicted Class</td><td>"+predClass+"</td></tr>");
            Application.predict = Integer.toString(predClass);
        }
        double accPercent = logisticPredict.accuracy(pred, yTest);
        String accStr = Double.toString(accPercent);
        output = output.concat("<tr><td>Training Accuracy</td><td>"+accStr+"%</td></tr>");
        Application.accuracy = accStr;
        output = output.concat("</table>");
        return ok(logisticResult.render());
    }

    public static Result backpropagation(){
        return ok(backpropagation.render());
    }

    public static Result outputBackpropagation(){
        Http.MultipartFormData body = request().body().asMultipartFormData();
        Http.MultipartFormData.FilePart datafile = body.getFile("dataFile");

        String trainFilename;
        if (datafile != null) {
            String fileName = datafile.getFilename();
            String contentType = datafile.getContentType();
            File file = datafile.getFile();
            file.renameTo(new File(uploadDir, fileName));
            trainFilename = uploadDir+"/"+fileName;
        } else {
            flash("error", "Missing file");
            return ok("Please upload file home page");
        }

        DynamicForm df = play.data.Form.form().bindFromRequest();
        String featureVal = df.get("featureArr");
        String classStr = df.get("classArr");
        String etaStr = df.get("eta");
        String iterStr = df.get("iter");
        String outputCol = df.get("outputCol");
        int outputColNo = Integer.parseInt(outputCol);

        double alpha = Double.valueOf(etaStr);
        int iter = Integer.valueOf(iterStr);

        boolean multiClass = false;

        featureVal = featureVal.replaceAll("\\s+","");
        String[] featureArr = featureVal.split(",");

        Matrix matrix = new Matrix();
        int[] classArr = matrix.strToIntArr(classStr);

        int outputlayersize = classArr.length;

                NNCostFunction nnCostFunction = new NNCostFunction();
        if(outputlayersize>2){
            nnCostFunction.setMulticlass(true);
            multiClass = true;
        }

        response().setContentType("text/html");
        String output = new String();
        output = output.concat("<h2>Backpropagation</h2><table style='width:40%'>");

        TextToArr textToArr = new TextToArr();
        String filepath = trainFilename;

        /***** 1. Learning Curve Start ******/
        LearningCurve learningCurve = new LearningCurve();
        learningCurve.initializeTrainTestcrossValidData(filepath, uploadDir+"backprop.json", ",", outputColNo);//initialize feature normalized train, test and cross validation data
        double[][] xTrain =learningCurve.getxTrain();
        double[][] xCross =learningCurve.getxCross();
        double[][] xTest = learningCurve.getxTest();

        if(featureArr.length<(xTrain[0].length)||featureArr.length>(xTrain[0].length)){
            return ok("No. of features "+featureArr.length+"!= no. of col in Train Data"+xTrain[0].length);
        }
        double[][] postX = new double[1][featureArr.length];
        for(int i=0; i<featureArr.length; i++){
            postX[0][i] = Double.parseDouble(featureArr[i]);
        }



        double[][] yTrain = learningCurve.getyTrain();
        double[][] yTest = learningCurve.getyTest();
        double[][] yCross = learningCurve.getyCross();

        yTrain = matrix.changeArrVal(yTrain, classArr);
        yTest = matrix.changeArrVal(yTest, classArr);
        yCross = matrix.changeArrVal(yCross, classArr);

        FeatureNormalize featureNormalize = new FeatureNormalize();

        double[][] X = featureNormalize.getFeatureNormalize(xTrain);
        postX = featureNormalize.predFeatureNormalize(postX,featureNormalize.getStd(), featureNormalize.getMean());

        xTest = featureNormalize.getFeatureNormalize(xTest);
//        double[][] X = xTrain;
        double[][] y = yTrain;

        /****** Learning Curve End *****/
        DebugInitializeWeights debugInitializeWeights = new DebugInitializeWeights();
        int inputLayerSize = X[0].length;
        int hiddenLayerSize = inputLayerSize/2;
//            int outputlayersize = 10;
//        int outputlayersize = 2;
        double lambda = 0.1;
        double[][] theta1 = debugInitializeWeights.createMatrix(hiddenLayerSize, inputLayerSize);
        double[][] theta2 = debugInitializeWeights.createMatrix(outputlayersize, hiddenLayerSize);
        Backwardpropopagation backwardpropopagation = new Backwardpropopagation();
        double[][] theta = backwardpropopagation.combineTheta(theta1, theta2);

        /******Check Neural Network Start*****/
//        CheckNNGradients checkNNGradients = new CheckNNGradients();
//        checkNNGradients.checkNeuralNetwork(lambda,inputLayerSize, hiddenLayerSize, outputlayersize, X, y, theta1, theta2, theta, multiClass);
//            checkNNGradients.checkNeuralNetwork(lambda);
        /******Check Neural Network End*****/

        /*****Gradient Descent Start*****/
        GradientDescentBackpropagation gradientDescentBackpropagation = new GradientDescentBackpropagation();
        double initialCost = nnCostFunction.cost(theta, inputLayerSize, hiddenLayerSize, outputlayersize, X, y, lambda);
        theta = gradientDescentBackpropagation.getGradient(theta, nnCostFunction.gettheta(),inputLayerSize,hiddenLayerSize,
                outputlayersize, X, y, lambda, alpha, iter, multiClass);

//            //Feb 2 start
//            double[] J_history = new double[iter];
//            for(int i=0; i<iter; i++){
//                J_history[i] = nnCostFunction.cost(theta, inputLayerSize, hiddenLayerSize, outputlayersize, X, y, lambda);
//                theta = nnCostFunction.gettheta();
//            }
//            //Feb 2 end
        /*****Gradient Descent End*****/
        double finalCost = nnCostFunction.cost(theta, inputLayerSize, hiddenLayerSize, outputlayersize, X, y, lambda);
        System.out.println("Neural Network cost: "+finalCost);
        output = output.concat("<tr><td>Cost</td><td>"+finalCost+"</td></tr>");
        Application.cost = Double.toString(finalCost);
//        double[][] pred = gradientDescentBackpropagation.getPred();
//        System.out.println("test"+Arrays.deepToString(pred));
        AccuracyFeedforward accuracyFeedforward = new AccuracyFeedforward();

        int[][] pred;
        if(classArr.length>2){
            pred = accuracyFeedforward.predict(xTest, gradientDescentBackpropagation.getTheta1(), gradientDescentBackpropagation.getTheta2(), 1);
            int predClass = accuracyFeedforward.predict(postX, gradientDescentBackpropagation.getTheta1(), gradientDescentBackpropagation.getTheta2(), 1)[0][0];

            predClass = matrix.predictClass(classArr, predClass);
            output = output.concat("<tr><td>Predicted Class</td><td>"+predClass+"</td></tr>");
            Application.predict = Integer.toString(predClass);
        }else{
            pred = accuracyFeedforward.predict(xTest, gradientDescentBackpropagation.getTheta1(), gradientDescentBackpropagation.getTheta2(), 0);
            int predClass = accuracyFeedforward.predict(postX, gradientDescentBackpropagation.getTheta1(), gradientDescentBackpropagation.getTheta2(), 0)[0][0];

            predClass = matrix.predictClass(classArr, predClass);
            output = output.concat("<tr><td>Predicted Class</td><td>"+predClass+"</td></tr>");
            Application.predict = Integer.toString(predClass);
        }
        LogisticPredict logisticPredict = new LogisticPredict();

        /******Gradient Descent Start*****/
//            LogisticGradientDescent gradientDescent = new LogisticGradientDescent();
//            Matrix matrix = new Matrix();
//            double[][] initalTheta = {{0}, {0}, {0}};
//            double alpha = 0.001;
//            int iter = 4000;
//            X = matrix.addColOfOnes(X);
//
//            double[][] gradTheta = gradientDescent.getGradientDescent(matrix.transpose(theta1), X, y, iter, alpha);
//            double gradCost = NNCostFunction.cost(gradTheta, inputLayerSize, hiddenLayerSize, outputlayersize, X, y, lambda);
//            System.out.println("Grad cost is "+gradCost+" normal cost is "+cost);
        /******Gradient Descent End*****/
        double accuracy = logisticPredict.accuracy(pred, yTest);
        output = output.concat("<tr><td>Training Accuracy</td><td>"+accuracy+"%</td></tr>");
        Application.accuracy = Double.toString(accuracy);
        output = output.concat("</table>");
        return ok(backpropagationResult.render());
    }

//    public static Result outputLogisticRegression() {
//        Http.MultipartFormData body = request().body().asMultipartFormData();
//        Http.MultipartFormData.FilePart datafile = body.getFile("dataFile");
//
//        String trainFilename;
//        if (datafile != null) {
//            String fileName = datafile.getFilename();
//            String contentType = datafile.getContentType();
//            File file = datafile.getFile();
//            file.renameTo(new File(uploadDir, fileName));
//            trainFilename = uploadDir+"/"+fileName;
//        } else {
//            flash("error", "Missing file");
//            return ok("Please upload file home page");
//        }
//
//        DynamicForm df = play.data.Form.form().bindFromRequest();
//
//        String featureVal = df.get("featureArr");
//        String postClass = df.get("logisticClass");
//        String etaStr = df.get("eta");
//        String iterStr = df.get("iter");
//        String outputCol = df.get("outputCol");
//        int outputColNo = Integer.parseInt(outputCol);
//
//        double alpha = Double.valueOf(etaStr);
//        int iter = Integer.valueOf(iterStr);
//
//        featureVal = featureVal.replaceAll("\\s+","");
//        String[] featureArr = featureVal.split(",");
//
//        double[][] postX = new double[1][featureArr.length];
//        for(int i=0; i<featureArr.length; i++){
//            postX[0][i] = Double.parseDouble(featureArr[i]);
//        }
//
//        response().setContentType("text/html");
//        String output = new String();
//        output = output.concat("<h2>Logistic Regression</h2><table style='width:40%'>");
//        String filepath = trainFilename;
//        String jsonfilePath = uploadDir+"/linearInput.json";
//
//        LearningCurve learningCurve = new LearningCurve();
//        String seperator = ",";
//        learningCurve.initializeTrainTestcrossValidData(filepath, jsonfilePath, seperator, outputColNo);//initialize feature normalized train, test and cross validation data
//        double[][] xTrain =learningCurve.getxTrain();
//        double[][] yTrain = learningCurve.getyTrain();
//        double[][] xTest = learningCurve.getxTest();
//        double[][] yTest = learningCurve.getyTest();
//        double[][] xCross = learningCurve.getxCross();
//        double[][] yCross = learningCurve.getyCross();
//
//        FeatureNormalize featureNormalize = learningCurve.getTrainfeatureNormalize();
//        xTest = featureNormalize.getFeatureNormalize(xTest);
//        yTest = featureNormalize.getFeatureNormalize(yTest);
//        xCross = featureNormalize.getFeatureNormalize(xCross);
//        yCross = featureNormalize.getFeatureNormalize(yCross);
//
//        if(featureArr.length+1!=xTrain[0].length){
//            return ok("number of features not equal to training data features"+featureArr.length+" "+xTrain[0].length);
//        }
//
//        final double[][] initalTheta = new double[xTrain[0].length][1];
////        double alpha = 0.1; //for bike data get the cost for classes nearly same and then we will have high accuracy
//        LogisticCost logisticCost = new LogisticCost();
//        double initialCost = logisticCost.getCost(xTrain, yTrain, initalTheta);
//        double[][] initialGrad = logisticCost.getGrad();
//        /*** Initial Cost End ***/
//
//        /***********iterations to find the optimum theta(gradient Descent) start**********/
////        int iter = 400; //60% accuracy and alpha 0.001
//        Matrix matrix = new Matrix();
//        LogisticGradientDescent gradientDescent = new LogisticGradientDescent();
//        double[][] gradTheta;
//        LogisticPredict logisticPredict = new LogisticPredict();
//        int[][] pred;
//
//        boolean multiClass = false;
//        if(postClass.equals("multi")){
//            String classVal = df.get("logisticClassArr");
//            classVal = classVal.replaceAll("\\s+","");
//            String[] classArr = classVal.split(",");
//
//            pred = new int[yTest.length][1];
//            ArrayList<double[][]> grad = new ArrayList();
//            ArrayList<double[][]> predArrL = new ArrayList();
//            ArrayList<Double> resultPred = new ArrayList();
//            for(int j=0; j<classArr.length; j++){
//                int yClass = Integer.valueOf(classArr[j]);
//                double[][] yVal = matrix.binaryClassfromMultiClass(yTrain, yClass);
//                gradTheta = gradientDescent.getGradientDescent(xTrain, yVal, iter, alpha);
//                grad.add(gradTheta);
//                double[][] predClass = logisticPredict.predictMultiClass(gradTheta, xTest, yClass);
//                predArrL.add(predClass);
//                resultPred.add(logisticPredict.predictMultiClass(gradTheta, postX, yClass)[0][0]);
//            }
//            for (int i = 0; i < yTest.length; i++) {
//                ArrayList<Double> maxArr = new ArrayList<Double>();
//                for(int j=0; j<classArr.length; j++){
//                    maxArr.add(predArrL.get(j)[i][0]);
//                }
//                pred[i][0] = Collections.max(maxArr);
//            }
//            double accPercent = logisticPredict.accuracy(pred, yTest);
//            String accStr = Double.toString(accPercent);
//            output = output.concat("<tr><td>Training Accuracy</td><td>"+accStr+"%</td></tr>");
//            output = output.concat("<tr><td>Predicted Class</td><td>"+Collections.max(resultPred)+"</td></tr>");
//        }else{
//            gradTheta = gradientDescent.getGradientDescent(xTrain, yTrain, iter, alpha);
//            double gradCost = gradientDescent.getCost();
//            System.out.println(xTrain.length+" train instances Cost: "+gradCost);
//            /***********Logistic Predictions Start**********/
//            int xTestRow = xTest.length;
//            int xTestCol = xTest[0].length;
//            double[][] input = new double[1][xTestCol]; //we don't need to add one here; This java program logic manages that one in LinearPredict.java
//            for(int i=0; i<input.length; i++){
//                for(int j=0; j<xTest[0].length; j++){
//                    input[0][j] = xTest[0][j];
//                }
//            }
//            double finalCost = logisticCost.getCost(xTrain, yTrain, gradTheta);
//            output = output.concat("<tr><td>Error</td><td>"+finalCost+"</td></tr>");
//
//            pred = logisticPredict.predict(gradTheta, xTest);
//            int[][] predResult = logisticPredict.predict(gradTheta, postX);
//            output = output.concat("<tr><td>Predicted Class</td><td>"+predResult[0][0]+"</td></tr>");
//            double accPercent = logisticPredict.accuracy(pred, yTest);
//            output = output.concat("<tr><td>Training Accuracy</td><td>"+accPercent+"%</td></tr>");
//        }
//        output = output.concat("</table>");
//        return ok(output);
//    }
}
