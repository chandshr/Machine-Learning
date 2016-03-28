package controllers.machineLearning.kmeans;

import controllers.machineLearning.all.*;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by shrestha on 12/30/2015.
 */

/**
 * Steps:
 * initialize Centroid
 * iterate 10 times and optimize the centorid
 * the optimized centroid are stored in array
 * assign machineLearning.all the datapoints to its nearest optimized array
 * points assignmed to a specific optimized centroid are stored in respective json file e.g firstCentroid.json: stores machineLearning.all the datapoints that are assigned to first optimized centroid
 */
public class Kmeans {

    public void get(){
        try {
            File currentDirectory = new File(new File(".").getAbsolutePath());
            String filePath = currentDirectory.getCanonicalPath()+"/src/main/resources/data/kmeans/X.txt";

            String seperator = "\\s+";
            TextToArr textToArr = new TextToArr();
            ArrayList<Point>  pointArr = textToArr.createPointArr(filePath, seperator);
            int noOfCentroid = 3;
            Point[] initialCentroid = {new Point(3, 3), new Point(6, 2), new Point(8, 5)};

            HashMap<Point, Point> closestCentroidOfX = new HashMap<Point, Point>();
            closestCentroidOfX = findClosestCentroid(pointArr, initialCentroid);
            System.out.println("Test closestCentroidOfX above");

            Point[] newHashCentroid = computeCentroids(closestCentroidOfX, initialCentroid);
            System.out.println("Test newHashCentroid value: ");

            Point[] optimizedCentroid = runKmeans(pointArr, initialCentroid, 10);
            System.out.println("Test optimizedCentroid value: ");

            HashMap<Point, Point> optimizedCentroidOfX = findClosestCentroid(pointArr, optimizedCentroid);
            Iterator iterator = optimizedCentroidOfX.entrySet().iterator();
            ArrayList<Double[]> firstCentroid = new ArrayList<Double[]>();
            ArrayList<Double[]> secondCentroid = new ArrayList<Double[]>();
            ArrayList<Double[]> thirdCentroid = new ArrayList<Double[]>();
            while (iterator.hasNext()) {
                HashMap.Entry<Point, Point> pair = (HashMap.Entry<Point, Point>)iterator.next();
                double hashValX = pair.getValue().getX();
                double hashValY = pair.getValue().getY();
                System.out.println(pair.getKey() + " = " + pair.getValue());
                Double[] temp = new Double[2];
                if(hashValX==optimizedCentroid[0].getX()&&hashValY==optimizedCentroid[0].getY()){
                    temp[0] = pair.getKey().getX();
                    temp[1] = pair.getKey().getY();
                    firstCentroid.add(temp);
                }else if(hashValX==optimizedCentroid[1].getX()&&hashValY==optimizedCentroid[1].getY()){
                    temp[0] = pair.getKey().getX();
                    temp[1] = pair.getKey().getY();
                    secondCentroid.add(temp);
                }else{
                    temp[0] = pair.getKey().getX();
                    temp[1] = pair.getKey().getY();
                    thirdCentroid.add(temp);
                }
            }
            System.out.println("Given input arranged according to optimized centroid in optimizedCentroidOfX");
            OutputJson json = new OutputJson();
            json.createJsonFile(optimizedCentroidOfX, "/src/main/resources/data/json/kmeansCentroidResult.json");
            json.createJsonFile(firstCentroid, "/src/main/resources/data/json/kmeansFirstCentroid.json");
            json.createJsonFile(secondCentroid, "/src/main/resources/data/json/kmeansSecondCentroid.json");
            json.createJsonFile(thirdCentroid, "/src/main/resources/data/json/kmeansThirdCentroid.json");
        }catch (IOException e) {
            e.printStackTrace(); //given input file not found exception
        }

    }

    public HashMap<Point, Point> findClosestCentroid(ArrayList<Point>  pointArr, Point[] centroidArr){
        HashMap<Point, Point> result = new HashMap<Point, Point>();
        double delta;
        double min;
        Point minCentroid = null;
        double x;
        double y;
        for(Point point : pointArr){
            double pointX = point.getX();
            double pointY = point.getY();

            //min initialization start
            x = (double) pointX-centroidArr[0].getX();
            y = (double) pointY-centroidArr[0].getY();
            min = (double) (x*x)+(y*y);
            minCentroid = centroidArr[0];
            //min initialization end

            //compare min with remaining centroids START
            for(int i=1; i<centroidArr.length; i++){
                x = (double) pointX-centroidArr[i].getX();
                y = (double) pointY-centroidArr[i].getY();
                delta = (double) (x*x)+(y*y);
                if(delta<min){
                    min = delta;
                    minCentroid = centroidArr[i];
                }
            }
            result.put(point, minCentroid);
            //compare min with remaining centroids END
        }
        return result;
    }

    public Point[] computeCentroids(HashMap<Point, Point> closestCentroidOfX, Point[] centroids){
        Set setOfCentroidAndX = closestCentroidOfX.entrySet();
        Iterator iterator = setOfCentroidAndX.iterator();

        HashMap<Point, Point> hashCentroid = new HashMap<Point, Point>();
        HashMap<Point, Integer> hashCentroidCount = new HashMap<Point, Integer>();
        HashMap<Point, Point> hashCentroidMean = new HashMap<Point, Point>();

        for(Point givenCentroid : centroids){
            hashCentroid.put(givenCentroid, new Point(0, 0));
            hashCentroidCount.put(givenCentroid, 0);
        }

        while (iterator.hasNext()){
            Map.Entry inputRow = (Map.Entry)iterator.next();
            Point inputPoint = (Point) inputRow.getKey();
            Point inputCentroid = (Point) inputRow.getValue();
            for(Point centroid : centroids){
                if(centroid.getX()==inputCentroid.getX()&&centroid.getY()==inputCentroid.getY()){
                    Point hashCentroidPoint = hashCentroid.get(centroid);
                    double xHashCentroidPoint = hashCentroidPoint.getX() + inputPoint.getX();
                    double yHashCentroidPoint = hashCentroidPoint.getY() + inputPoint.getY();
                    hashCentroidPoint = new Point(xHashCentroidPoint, yHashCentroidPoint);
                    hashCentroid.put(centroid, hashCentroidPoint);

                    int count = hashCentroidCount.get(centroid);
                    hashCentroidCount.put(centroid, count+1);
                }
            }
        }

        //result computation
        Point[] resultCentroid = new Point[centroids.length];
        int i=0;
        for(Point givenCentroid : centroids){
            Point point = hashCentroid.get(givenCentroid);
            int num = hashCentroidCount.get(givenCentroid);
            double pointX = (double) point.getX()/num;
            double pointY = (double) point.getY()/num;
            Point newCentroid = new Point(pointX, pointY);
            hashCentroidMean.put(givenCentroid, newCentroid);
            resultCentroid[i] = newCentroid;
            i++;
        }

        return  resultCentroid;
    }

    public Point[] runKmeans(ArrayList<Point>  pointArr, Point[] centroidArr, int iter){
        for(int i=0; i<iter; i++){
            HashMap<Point, Point> closestCentroidOfX = findClosestCentroid(pointArr, centroidArr);
            centroidArr = computeCentroids(closestCentroidOfX, centroidArr);
        }
        return centroidArr;
    }
}
