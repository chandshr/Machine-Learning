package controllers.machineLearning.all;

import controllers.machineLearning.kmeans.Point;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by shresc2 on 11/10/2015.
 */
public class TextToArr {

    public  ArrayList<ArrayList<Double>> convert(String filePath, String seperator){
        BufferedReader inputStream = null;
        ArrayList<ArrayList<Double>> output = null;
        try {
            System.out.println(filePath);
            File inputFile = new File(filePath);
            inputStream = new BufferedReader(new FileReader(inputFile));
            String strLine;
            Scanner scanner = new Scanner(inputStream);
            output = new ArrayList<ArrayList<Double>>();
            scanner.nextLine();//skip header of .csv file
            while (scanner.hasNextLine()) {
                ArrayList<Double> wordArrL = new ArrayList<Double>();
                strLine = scanner.nextLine();
                strLine = strLine.replaceAll("\\s+","");
                String[] words = strLine.split(seperator);
                ArrayList<Double> dataArrL = new ArrayList<Double>();
                for(int i=0; i<words.length; i++){
                    String dataStr = words[i];
                    wordArrL.add(Double.parseDouble(dataStr));
                }
                output.add(wordArrL);
            }
            inputStream.close();
        }catch (IOException e){
            System.out.println (e.toString());
            System.out.println("Could not find file " + inputStream);
        }
        return output;
    }

    public double[][] datatoArr(ArrayList<ArrayList<Double>> data){
        int row = data.size();
        int col = data.get(0).size();
        double[][] result = new double[row][col];
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                result[i][j] = data.get(i).get(j);
            }
        }
        return result;
    }

    public ArrayList<Point> createPointArr(String filePath, String seprator) throws IOException{
        BufferedReader inputStream = null;
        ArrayList<Point> result = new ArrayList<Point>();
        try {
            File currentDirectory = new File(new File(".").getAbsolutePath());
            inputStream = new BufferedReader(new FileReader(filePath));
            Scanner scanner = new Scanner(inputStream);

            String strLine;
            Point point = null;

            while (scanner.hasNextLine()) {
                strLine = scanner.nextLine();
                String[] words = strLine.split(seprator);
                for(int i=0; i<words.length; i++){
                    String x = words[0];
                    String y = words[1];
                    point = new Point(Double.parseDouble(x), Double.parseDouble(y));
                }
                result.add(point);
            }
        }finally {
            if (inputStream != null) {
                inputStream.close();
            }
        }
        return result;
    }

    /**
     * convert input file of txt format to double[][] array
     * @return
     */
    //TODO: use classpath TRIED
    public double[][] convertInputFiletoInputArr(String filepath, String strSeperator){
        System.out.println(filepath);
        double[][] dataArr;
        TextToArr textToArr = new TextToArr();
        ArrayList<ArrayList<Double>> data = textToArr.convert(filepath, strSeperator);
        dataArr = textToArr.datatoArr(data);
        return dataArr;
    }
}
