package controllers.machineLearning.GraphPlot;

import javax.swing.*;

/**
 * Created by shrestha on 1/4/2016.
 */
public class Graph {
    public void get(){
        Points points = new Points();
        JFrame frame = new JFrame("Points");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(points);
        frame.setSize(250, 200);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
