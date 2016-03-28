package controllers.machineLearning.GraphPlot;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

/**
 * Created by shrestha on 1/4/2016.
 */
public class Points extends JPanel{
    public void paintComponent(Graphics g) {
        super.paintComponent(g);

        Graphics2D g2d = (Graphics2D) g;

        g2d.setColor(Color.red);

        for (int i = 0; i <= 100000; i++) {
            Dimension size = getSize();
            int w = size.width ;
            int h = size.height;

            Random r = new Random();
            int x = Math.abs(r.nextInt()) % w;
            int y = Math.abs(r.nextInt()) % h;
            g2d.drawLine(x, y, x, y);
            g2d.drawString("+", x, y);
        }
    }

    public void plotscatterPoint(Graphics g, String filePath, String seperator) throws IOException {
        super.paintComponent(g);
        Graphics2D graphic2D = (Graphics2D) g;

        BufferedReader inputStream = null;
        ArrayList<ArrayList<Double>> output = null;
        try {
            File currentDirectory = new File(new File(".").getAbsolutePath());

            inputStream = new BufferedReader(new FileReader(filePath));
            String strLine;
            Scanner scanner = new Scanner(inputStream);
            output = new ArrayList<ArrayList<Double>>();
            while (scanner.hasNextLine()) {
                strLine = scanner.nextLine();
                String[] words = strLine.split(seperator);
//                double x = Double.parseDouble(words[0]);
//                double y = Double.parseDouble(words[1]);
                int x = Integer.parseInt(words[0]);
                int y = Integer.parseInt(words[1]);
                if(words[2]=="1"){
                    graphic2D.setColor(Color.green);
                    graphic2D.drawString("+", x, y);
                }

            }
        }finally {
            if (inputStream != null) {
                inputStream.close();
            }
        }
    }
}
