package machine_learning_algorithms;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;

public class Random_Guess_Algorithm {

    /**
     * This Class generates random predictions for each test data point
     * and evaluates the accuracy of these random predictions.
     */

    public void run() throws FileNotFoundException {
        List<DataPoint> Dataset_1 = loadData(AlgorithmRunner.Dataset_1);
        List<DataPoint> Dataset_2 = loadData(AlgorithmRunner.Dataset_2);

        // Test Data set 1
        System.out.println("Testing on Dataset 1:");
        double fold1Accuracy = evaluateAccuracy(Dataset_1);

        // Test Data set 2
        System.out.println("Testing on Dataset 2:");
        double fold2Accuracy = evaluateAccuracy(Dataset_2);

        // List out the results for both folds
        System.out.println("\nAlgorithm: Random Guess");
        System.out.println("Summary of Results:");
        System.out.printf("Fold 1 Accuracy: %.2f%%\n", fold1Accuracy);
        System.out.printf("Fold 2 Accuracy: %.2f%%\n", fold2Accuracy);
    }

    // Method to evaluate accuracy for a given dataset and return the accuracy
    private double evaluateAccuracy(List<DataPoint> testData) {
        int correctPredictions = 0;
        Random random = new Random();

        // Initialize the confusion matrix (10x10 for digits 0-9)
        int[][] confusionMatrix = new int[10][10];

        // Loop through each test data point to make a random guess
        for (int index = 0; index < testData.size(); index++) {
            DataPoint testPoint = testData.get(index);
            // Generate a random digit between 0 and 9
            int predictedLabel = random.nextInt(10);
            int actualLabel = Integer.parseInt(testPoint.label);

            // Update confusion matrix
            confusionMatrix[actualLabel][predictedLabel]++;

            // Check if the predicted label matches the actual label
            if (actualLabel == predictedLabel) {
                correctPredictions++;
                if (AlgorithmRunner.Print_Details) {
                    System.out.println("\u2714 Correct! Actual: " + actualLabel + ", Predicted: " + predictedLabel);
                }
            } else if (AlgorithmRunner.Print_Details) {
                System.out.println("\u274C Incorrect! Actual: " + actualLabel + ", Predicted: " + predictedLabel);
            }
        }

        // Print the confusion matrix
        System.out.println("\nConfusion Matrix:");
        System.out.print("       ");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%7d", i);
        }
        System.out.println();
        for (int i = 0; i < 10; i++) {
            System.out.printf("%7d", i);
            for (int j = 0; j < 10; j++) {
                System.out.printf("%7d", confusionMatrix[i][j]);
            }
            System.out.println();
        }

        // Calculate and return the final accuracy
        double accuracy = (double) correctPredictions / testData.size() * 100;
        return accuracy;
    }

    // Method to load data from the file
    private List<DataPoint> loadData(String filePath) throws FileNotFoundException {
        List<DataPoint> dataPoints = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] dataLineValues = line.split(",");
                int columnCount = dataLineValues.length;
                List<Double> features = new ArrayList<>();
                for (int columnIndex = 0; columnIndex < columnCount - 1; columnIndex++) {
                    features.add(Double.parseDouble(dataLineValues[columnIndex]));
                }
                // Last column is the label
                String label = dataLineValues[columnCount - 1];
                dataPoints.add(new DataPoint(features, label));
            }
        }
        return dataPoints;
    }

    // Constructor to initialize a data point.
    static class DataPoint {
        List<Double> features;
        String label;

        DataPoint(List<Double> features, String label) {
            this.features = features;
            this.label = label;
        }
    }
}
