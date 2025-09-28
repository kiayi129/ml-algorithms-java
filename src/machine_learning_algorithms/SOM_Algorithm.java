package machine_learning_algorithms;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SOM_Algorithm {
	
    /**
     * This Class runs the Self-Organizing Map Algorithm with training and test data sets.
     */
	
	public void run() throws FileNotFoundException {
        // Load Dataset 1 and Dataset 2
        double[][] trainData1 = loadData(AlgorithmRunner.Dataset_1);
        int[] trainLabels1 = loadLabels(AlgorithmRunner.Dataset_1);

        double[][] trainData2 = loadData(AlgorithmRunner.Dataset_2);
        int[] trainLabels2 = loadLabels(AlgorithmRunner.Dataset_2);

        // Initialize the SOM algorithm with appropriate grid size and input dimension (e.g., 64 for 64 features)
        SOM_Algorithm som1 = new SOM_Algorithm(10, 10, 64);  // Example grid size and input dimension
        SOM_Algorithm som2 = new SOM_Algorithm(10, 10, 64);  // Another SOM instance for the second fold

        // Test on Dataset 1 (Train on Dataset 1, Test on Dataset 2)
        System.out.println("Testing on Dataset 1 (Train on Dataset 1, Test on Dataset 2):");
        som1.train(trainData1, 10000);  // Train on Dataset 1 (10000 iterations for example)
        int[][] nodeLabels1 = som1.assignLabelsToNodes(trainData1, trainLabels1);  // Assign labels based on training data
        double fold1Accuracy = som1.test(trainData2, trainLabels2, nodeLabels1);  // Test on Dataset 2
        System.out.printf("Fold 1 Accuracy: %.2f%%", fold1Accuracy);

        // Test on Dataset 2 (Train on Dataset 2, Test on Dataset 1)
        System.out.println("\n\nTesting on Dataset 2 (Train on Dataset 2, Test on Dataset 1):");
        som2.train(trainData2, 10000);  // Train on Dataset 2 (10000 iterations for example)
        int[][] nodeLabels2 = som2.assignLabelsToNodes(trainData2, trainLabels2);  // Assign labels based on training data
        double fold2Accuracy = som2.test(trainData1, trainLabels1, nodeLabels2);  // Test on Dataset 1
        System.out.printf("Fold 2 Accuracy: %.2f%%\n", fold2Accuracy);

        // List out the results for both folds
        System.out.println("\nAlgorithm: Self-Organizing Map");
        System.out.println("Summary of Results:");
        System.out.printf("Fold 1 Accuracy: %.2f%%", fold1Accuracy);
        System.out.printf("\nFold 2 Accuracy: %.2f%%", fold2Accuracy);
    }
	
    private int gridWidth;
    private int gridHeight;
    private int inputDim;
    private double[][][] weights;
    private double learningRate;
    private double neighborhoodRadius;

    public SOM_Algorithm(int gridWidth, int gridHeight, int inputDim) {
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;
        this.inputDim = inputDim;
        this.weights = new double[gridWidth][gridHeight][inputDim];
        this.learningRate = 0.3;
        this.neighborhoodRadius = Math.max(gridWidth, gridHeight) / 2.0;
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < gridWidth; i++) {
            for (int j = 0; j < gridHeight; j++) {
                for (int k = 0; k < inputDim; k++) {
                    weights[i][j][k] = rand.nextDouble();  // Initialize weights randomly
                }
            }
        }
    }

    private double euclideanDistance(double[] input, double[] weights) {
        double sum = 0;
        for (int i = 0; i < input.length; i++) {
            sum += Math.pow(input[i] - weights[i], 2);
        }
        return Math.sqrt(sum);
    }

    private int[] findBMU(double[] input) {
        int[] bmu = new int[2];
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < gridWidth; i++) {
            for (int j = 0; j < gridHeight; j++) {
                double distance = euclideanDistance(input, weights[i][j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    bmu[0] = i;
                    bmu[1] = j;
                }
            }
        }
        return bmu;
    }

    private void updateWeights(double[] input, int bmuX, int bmuY, double learningRate, double radius) {
        for (int i = 0; i < gridWidth; i++) {
            for (int j = 0; j < gridHeight; j++) {
                double distance = Math.sqrt(Math.pow(bmuX - i, 2) + Math.pow(bmuY - j, 2));
                if (distance < radius) {
                    double influence = Math.exp(-distance / (2 * Math.pow(radius, 2)));
                    for (int k = 0; k < inputDim; k++) {
                        weights[i][j][k] += influence * learningRate * (input[k] - weights[i][j][k]);
                    }
                }
            }
        }
    }

    private void train(double[][] data, int numIterations) {
        initializeWeights();  // Initialize weights using the data

        // Train the SOM
        for (int iter = 0; iter < numIterations; iter++) {
            Random rand = new Random();
            double[] input = data[rand.nextInt(data.length)];  // Randomly pick a training sample

            // Ensure that input size matches weight size
            if (input.length != inputDim) {
                throw new IllegalArgumentException("Input dimension mismatch: expected " + inputDim + " but got " + input.length);
            }

            int[] bmu = findBMU(input);  // Find the Best Matching Unit
            updateWeights(input, bmu[0], bmu[1], learningRate, neighborhoodRadius);  // Update the SOM weights

            // Slowly decay learning rate and neighborhood radius
            learningRate = learningRate * (1 - (double) iter / numIterations);
            neighborhoodRadius = Math.max(1, neighborhoodRadius * (1 - (double) iter / numIterations));
        }
    }

    public double test(double[][] testData, int[] testLabels, int[][] nodeLabels) {
        int correct = 0;
        int[][] confusionMatrix = new int[10][10]; // 10 classes (digits 0-9)

        for (int i = 0; i < testData.length; i++) {
            int[] bmu = findBMU(testData[i]); // Find the BMU for the test data
            int predictedLabel = nodeLabels[bmu[0]][bmu[1]]; // Get the predicted label for the BMU
            int actualLabel = testLabels[i]; // Actual label

            // Update confusion matrix
            confusionMatrix[actualLabel][predictedLabel]++;

            if (predictedLabel == actualLabel) { // Compare with the actual label
                correct++;
            }
        }

        // Print Confusion Matrix with labels 0-9
        System.out.println("Confusion Matrix:");
        System.out.print("       ");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%7d", i); // Print column labels
        }
        System.out.println();
        for (int i = 0; i < 10; i++) {
            System.out.printf("%7d", i); // Print row labels
            for (int j = 0; j < 10; j++) {
                System.out.printf("%7d", confusionMatrix[i][j]);
            }
            System.out.println();
        }

        return (double) correct / testData.length * 100; // Return accuracy as percentage
    }

    private int[][] assignLabelsToNodes(double[][] data, int[] labels) {
        // Create a 2D array to store the label of each node in the SOM grid
        int[][] nodeLabels = new int[gridWidth][gridHeight];

        // Initialize the count of labels per node (for majority vote)
        int[][][] labelCount = new int[gridWidth][gridHeight][10]; // Assuming 10 possible labels

        // Map each training sample to the corresponding BMU and update label count
        for (int i = 0; i < data.length; i++) {
            double[] input = data[i];
            int label = labels[i];
            int[] bmu = findBMU(input);  // Find the BMU for this input
            labelCount[bmu[0]][bmu[1]][label]++;  // Increment the count for the corresponding label
        }

        // Assign the most frequent label to each node in the grid
        for (int i = 0; i < gridWidth; i++) {
            for (int j = 0; j < gridHeight; j++) {
                int maxCount = -1;
                int assignedLabel = -1;
                for (int l = 0; l < 10; l++) {
                    if (labelCount[i][j][l] > maxCount) {
                        maxCount = labelCount[i][j][l];
                        assignedLabel = l;
                    }
                }
                nodeLabels[i][j] = assignedLabel;  // Assign the label to the node
            }
        }

        return nodeLabels;
    }

    public static double[][] loadData(String filePath) throws FileNotFoundException {
        List<double[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isHeader = true;  // Flag to skip the header row
            while ((line = br.readLine()) != null) {
                if (isHeader) {
                    isHeader = false;  // Skip the first row (header)
                    continue;
                }

                // Split the line by commas
                String[] values = line.split(",");

                // Check if the row has 65 columns (64 features and 1 label)
                if (values.length != 65) {
                    System.out.println("Warning: Row with incorrect number of features, expected 65.");
                    continue;  // Skip rows with incorrect number of columns
                }

                // Create an array for the 64 features (ignore the label column)
                double[] features = new double[64];
                for (int i = 0; i < 64; i++) {
                    features[i] = Double.parseDouble(values[i]);  // Use the first 64 columns as features
                }

                // Add the features to the data list
                data.add(features);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return data.toArray(new double[data.size()][]);
    }

    // Helper function to load labels
    private int[] loadLabels(String filePath) throws FileNotFoundException {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isHeader = true;
            while ((line = br.readLine()) != null) {
                if (isHeader) {
                    isHeader = false;
                    continue;
                }
                String[] values = line.split(",");
                int label = Integer.parseInt(values[64]);  // Assuming label is in the last column
                labels.add(label);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labels.stream().mapToInt(i -> i).toArray();
    }
}
