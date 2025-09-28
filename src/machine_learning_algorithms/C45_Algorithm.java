package machine_learning_algorithms;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class C45_Algorithm {

    /**
     * This Class runs the C4.5 Algorithm with training and test data sets.
     */
    
    public void run() {
        List<String[]> dataset1 = loadData(AlgorithmRunner.Dataset_1);
        List<String[]> dataset2 = loadData(AlgorithmRunner.Dataset_2);
        System.out.printf("Dataset 1 Size: %d rows, %d columns%n", dataset1.size(), dataset1.get(0).length);
        System.out.printf("Dataset 2 Size: %d rows, %d columns%n", dataset2.size(), dataset2.get(0).length);
        int targetIndex = dataset1.get(0).length - 1;

        // First fold: Train on Dataset 1, Test on Dataset 2
        System.out.println("\nRunning Fold 1 (Training on Dataset 1, Testing on Dataset 2):");
        List<Integer> attributes = new ArrayList<>();
        for (int i = 0; i < targetIndex; i++) {
            attributes.add(i);
        }
        Node tree = buildTree(dataset1, attributes, targetIndex);
        double[][] confusionMatrixFold1 = evaluateModel(tree, dataset2, targetIndex);
        printConfusionMatrix(confusionMatrixFold1);
        double accuracyFold1 = calculateAccuracy(confusionMatrixFold1);
        System.out.printf("Accuracy (Fold 1): %.2f%%\n", accuracyFold1 * 100);

        // Second fold: Train on Dataset 2, Test on Dataset 1
        System.out.println("\nRunning Fold 2 (Training on Dataset 2, Testing on Dataset 1):");
        tree = buildTree(dataset2, attributes, targetIndex);
        double[][] confusionMatrixFold2 = evaluateModel(tree, dataset1, targetIndex);
        printConfusionMatrix(confusionMatrixFold2);
        double accuracyFold2 = calculateAccuracy(confusionMatrixFold2);
        System.out.printf("Accuracy (Fold 2): %.2f%%\n", accuracyFold2 * 100);
        
        // List out the results for both folds
        System.out.println("\nAlgorithm: C4.5");
        System.out.println("Summary of Results:");
        System.out.printf("Fold 1 Accuracy: %.2f%%\n", accuracyFold1 * 100);
        System.out.printf("Fold 2 Accuracy: %.2f%%\n", accuracyFold2 * 100);
    }

    // Loads data from a CSV file
    public static List<String[]> loadData(String datasetFile) {
        List<String[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(datasetFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                data.add(line.split(","));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }
    
    // Prints the confusion matrix
    public void printConfusionMatrix(double[][] confusionMatrix) {
        System.out.println("\nConfusion Matrix:");
        
        // Print column headers (0 to 9 for class labels)
        System.out.print("       ");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%5d ", i);  // Adjust column width as needed
        }
        System.out.println();

        // Print the matrix rows with row labels (0 to 9 for class labels)
        for (int i = 0; i < confusionMatrix.length; i++) {
            System.out.printf("%2d    ", i);  // Print row label
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                System.out.printf("%5.0f ", confusionMatrix[i][j]);  // Print matrix values
            }
            System.out.println();
        }
    }
    
    // Calculate accuracy based on the confusion matrix
    public double calculateAccuracy(double[][] confusionMatrix) {
        double correctPredictions = 0;
        double totalPredictions = 0;
        for (int i = 0; i < confusionMatrix.length; i++) {
            correctPredictions += confusionMatrix[i][i];
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                totalPredictions += confusionMatrix[i][j];
            }
        }
        return correctPredictions / totalPredictions;
    }
    
    // Class representing a node in the decision tree
    static class Node {
        String attribute;
        String label;
        String majorityClass;
        Map<String, Node> children = new HashMap<>();
    }

    // Builds the decision tree recursively
    private Node buildTree(List<String[]> dataset, List<Integer> availableAttributes, int targetIndex) {
        if (dataset.isEmpty()) {
            return null;
        }

        // Check if all instances have the same label
        Set<String> labels = new HashSet<>();
        for (String[] instance : dataset) {
            labels.add(instance[targetIndex]);
        }
        if (labels.size() == 1) {
            Node leaf = new Node();
            leaf.label = labels.iterator().next();
            return leaf;
        }

        // Check if no attributes are available for splitting
        if (availableAttributes.isEmpty()) {
            Node leaf = new Node();
            leaf.label = findMajorityClass(dataset, targetIndex);
            return leaf;
        }

        // Find the best attribute to split on
        int bestAttribute = selectBestAttribute(dataset, availableAttributes, targetIndex);
        Node node = new Node();
        node.attribute = "Attribute_" + bestAttribute;
        node.majorityClass = findMajorityClass(dataset, targetIndex);

        // Split dataset into subsets based on the best attribute
        Map<String, List<String[]>> splitData = new HashMap<>();
        for (String[] instance : dataset) {
            String attributeValue = instance[bestAttribute];
            splitData.computeIfAbsent(attributeValue, k -> new ArrayList<>()).add(instance);
        }

        // Recurse for each subset
        List<Integer> remainingAttributes = new ArrayList<>(availableAttributes);
        remainingAttributes.remove(Integer.valueOf(bestAttribute));
        for (Map.Entry<String, List<String[]>> entry : splitData.entrySet()) {
            String attributeValue = entry.getKey();
            List<String[]> subset = entry.getValue();

            if (subset.isEmpty()) {
                Node leaf = new Node();
                leaf.label = findMajorityClass(dataset, targetIndex);
                node.children.put(attributeValue, leaf);
            } else {
                node.children.put(attributeValue, buildTree(subset, remainingAttributes, targetIndex));
            }
        }

        return node;
    }

    // Classifies a single instance using the decision tree
    private String classifyInstance(Node node, String[] instance) {
        if (node.label != null) {
            return node.label;
        }

        int attributeIndex = Integer.parseInt(node.attribute.split("_")[1]);
        String attributeValue = instance[attributeIndex];
        Node child = node.children.get(attributeValue);

        if (child == null) {
            System.out.printf("Fallback to majority class: %s at node splitting on '%s' (value: '%s')%n",
                    node.majorityClass, node.attribute, attributeValue);
            return node.majorityClass;
        }

        return classifyInstance(child, instance);
    }

    // Evaluates the decision tree on a test dataset and returns the confusion matrix
    public double[][] evaluateModel(Node root, List<String[]> testData, int targetIndex) {
        int numClasses = 10; // Assuming digits 0-9 (for a total of 10 classes)
        double[][] confusionMatrix = new double[numClasses][numClasses]; // Initialize confusion matrix

        for (String[] instance : testData) {
            String predicted = classifyInstance(root, instance);
            String actual = instance[targetIndex];

            int predictedLabel = Integer.parseInt(predicted);
            int actualLabel = Integer.parseInt(actual);

            // Update the confusion matrix based on predicted and actual values
            confusionMatrix[actualLabel][predictedLabel]++;
        }

        return confusionMatrix;
    }

    // Finds the majority class in a dataset
    private String findMajorityClass(List<String[]> dataset, int targetIndex) {
        Map<String, Integer> classCounts = new HashMap<>();
        for (String[] instance : dataset) {
            String targetValue = instance[targetIndex];
            classCounts.put(targetValue, classCounts.getOrDefault(targetValue, 0) + 1);
        }

        return classCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get()
                .getKey();
    }

    // Selects the best attribute for splitting using information gain
    private int selectBestAttribute(List<String[]> dataset, List<Integer> availableAttributes, int targetIndex) {
        double bestGain = Double.NEGATIVE_INFINITY;
        int bestAttribute = -1;

        for (int attribute : availableAttributes) {
            double gain = calculateInformationGain(dataset, attribute, targetIndex);
            if (gain > bestGain) {
                bestGain = gain;
                bestAttribute = attribute;
            }
        }
        return bestAttribute;
    }

    // Calculates information gain for a specific attribute
    private double calculateInformationGain(List<String[]> dataset, int attributeIndex, int targetIndex) {
        double entropyBefore = calculateEntropy(dataset, targetIndex);
        Map<String, List<String[]>> splitData = new HashMap<>();
        for (String[] instance : dataset) {
            String attributeValue = instance[attributeIndex];
            splitData.computeIfAbsent(attributeValue, k -> new ArrayList<>()).add(instance);
        }

        double entropyAfter = 0.0;
        for (List<String[]> subset : splitData.values()) {
            double subsetProbability = (double) subset.size() / dataset.size();
            entropyAfter += subsetProbability * calculateEntropy(subset, targetIndex);
        }

        return entropyBefore - entropyAfter;
    }

    // Calculates entropy of a dataset based on the target attribute
    private double calculateEntropy(List<String[]> dataset, int targetIndex) {
        Map<String, Integer> classCounts = new HashMap<>();
        for (String[] instance : dataset) {
            String targetValue = instance[targetIndex];
            classCounts.put(targetValue, classCounts.getOrDefault(targetValue, 0) + 1);
        }

        double entropy = 0.0;
        for (int count : classCounts.values()) {
            double probability = (double) count / dataset.size();
            entropy -= probability * Math.log(probability) / Math.log(2);
        }
        return entropy;
    }
}
