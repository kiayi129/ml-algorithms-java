package machine_learning_algorithms;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Scanner;

public class KNN_Algorithm {

    /**
     * This Class runs the K-Nearest Neighbors Algorithm with training and test data sets.
     */
    
    public void run() throws FileNotFoundException {
        List<DataPoint> Dataset_1 = loadData(AlgorithmRunner.Dataset_1);
        List<DataPoint> Dataset_2 = loadData(AlgorithmRunner.Dataset_2);

        // First fold: Train with original training and test with original test data
        System.out.println("Running Fold 1 (Training on Dataset 1, Testing on Dataset 2):");
        double fold1Accuracy = performKNN(Dataset_1, Dataset_2);

        // Second fold: Swap training and test data (Train with Test Data and Test with Training Data)
        System.out.println("Running Fold 2 (Training on Dataset 2, Testing on Dataset 1):");
        double fold2Accuracy = performKNN(Dataset_2, Dataset_1);

        // List out the results for both folds
        System.out.println("\nAlgorithm: K-Nearest Neighbors");
        System.out.println("Summary of Results:");
        System.out.printf("Fold 1 Accuracy: %.2f%%\n", fold1Accuracy);
        System.out.printf("Fold 2 Accuracy: %.2f%%\n", fold2Accuracy);
    }

    // Method to perform KNN classification
    private double performKNN(List<DataPoint> trainingData, List<DataPoint> testData) {
        // Determine unique labels for confusion matrix
        List<String> labels = determineUniqueLabels(trainingData, testData);
        int[][] confusionMatrix = new int[labels.size()][labels.size()];
        int correctPredictions = 0;

        // Iterate through each test point for classification
        for (DataPoint testPoint : testData) {
            String predictedLabel = classify(testPoint, trainingData);

            // Get indices for confusion matrix
            int actualIndex = labels.indexOf(testPoint.label);
            int predictedIndex = labels.indexOf(predictedLabel);

            // Update confusion matrix
            confusionMatrix[actualIndex][predictedIndex]++;

            // Check if the prediction matches the actual label
            if (testPoint.label.equals(predictedLabel)) {
                correctPredictions++;
                if (AlgorithmRunner.Print_Details) {
                    System.out.println("✔ Correct! Actual: " + testPoint.label + ", Predicted: " + predictedLabel);
                }
            } else if (AlgorithmRunner.Print_Details) {
                System.out.println("❌ Incorrect! Actual: " + testPoint.label + ", Predicted: " + predictedLabel);
            }
        }

        // Print confusion matrix
        System.out.println("\nConfusion Matrix:");
        printConfusionMatrix(confusionMatrix, labels);

        // Calculate and return accuracy
        double accuracyPercentage = (double) correctPredictions / testData.size() * 100;
        return accuracyPercentage;
    }

    // Method to determine unique labels in training and test data
    private List<String> determineUniqueLabels(List<DataPoint> trainingData, List<DataPoint> testData) {
        Map<String, Boolean> labelSet = new HashMap<>();
        for (DataPoint point : trainingData) {
            labelSet.put(point.label, true);
        }
        for (DataPoint point : testData) {
            labelSet.put(point.label, true);
        }
        return new ArrayList<>(labelSet.keySet());
    }

    // Method to print the confusion matrix
    private void printConfusionMatrix(int[][] confusionMatrix, List<String> labels) {
        System.out.print("       ");
        for (String label : labels) {
            System.out.printf("%-8s", label);
        }
        System.out.println();

        for (int i = 0; i < confusionMatrix.length; i++) {
            System.out.printf("%-8s", labels.get(i));
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                System.out.printf("%-8d", confusionMatrix[i][j]);
            }
            System.out.println();
        }
    }

    // Load data from CSV file
    private List<DataPoint> loadData(String filePath) throws FileNotFoundException {
        List<DataPoint> dataPoints = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] lineComponents = line.split(",");
                int featureCount = lineComponents.length;
                List<Double> features = new ArrayList<>();
                for (int i = 0; i < featureCount - 1; i++) {
                    features.add(Double.parseDouble(lineComponents[i]));
                }
                String label = lineComponents[featureCount - 1];
                dataPoints.add(new DataPoint(features, label));
            }
        }
        return dataPoints;
    }

    // Classify a test point based on training data
    private String classify(DataPoint testPoint, List<DataPoint> trainingData) {
        // Priority queue to store the neighbors sorted by distance
        PriorityQueue<Result> result = new PriorityQueue<>(Comparator.comparingDouble(n -> n.distance));

        // Calculate the distance for each training data point and add to the queue
        for (DataPoint trainPoint : trainingData) {
            double distance = calculateEuclideanDistance(testPoint.features, trainPoint.features);
            result.add(new Result(trainPoint.label, distance));
        }

        // Use majority voting to determine the predicted label
        Map<String, Integer> labelCount = new HashMap<>();
        for (int i = 0; i < AlgorithmRunner.K_Value && !result.isEmpty(); i++) {
            Result final_result = result.poll();
            labelCount.put(final_result.label, labelCount.getOrDefault(final_result.label, 0) + 1);
        }

        // Return the label with the highest count
        return labelCount.entrySet().stream()
                .max(Comparator.comparingInt(Map.Entry::getValue))
                .orElseThrow()
                .getKey();
    }

    // Calculate the Euclidean distance between two points
    private double calculateEuclideanDistance(List<Double> Point_A, List<Double> Point_B) {
        double squaredDistanceSum = 0.0;
        for (int featureIndex = 0; featureIndex < Point_A.size(); featureIndex++) {
            squaredDistanceSum += Math.pow(Point_A.get(featureIndex) - Point_B.get(featureIndex), 2);
        }
        return Math.sqrt(squaredDistanceSum);
    }

    // Inner class for representing a data point
    static class DataPoint {
        List<Double> features;
        String label;

        DataPoint(List<Double> features, String label) {
            this.features = features;
            this.label = label;
        }
    }

    // Inner class for storing the result
    static class Result {
        String label;
        double distance;

        Result(String label, double distance) {
            this.label = label;
            this.distance = distance;
        }
    }
}
