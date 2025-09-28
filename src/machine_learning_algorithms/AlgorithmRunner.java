package machine_learning_algorithms;

import java.io.FileNotFoundException;

public class AlgorithmRunner {

    // Default settings
    public static final String Dataset_1 = "./data/dataset1.csv"; 
    public static final String Dataset_2 = "./data/dataset2.csv"; 
    public final static boolean Print_Details = true; // If set to true will print the guess and original labels
    
    // K-nearest neighbors settings
    public final static boolean KNN_Algorithm = false; // If set to true, it will run KNN algorithm
    public final static int K_Value = 1; // Value for the amount of the numbers to check for (Best results with K = 1)
    
    // Random Guess settings
    public final static boolean Random_Guess_Algorithm = false; // If set to true, it will run Random Guess algorithm
    
    // SOM Self-Organizing Maps settings
    public final static boolean SOM_Algorithm = false; // If set to true, it will run SOM algorithm

    // C4.5 Decision Tree settings
    public final static boolean C45_Algorithm = true; // If set to true, it will run MLP algorithm

    public static void main(String[] args) {
    	// This will run the K-Nearest Neighbor Algorithm 
        if (KNN_Algorithm) {
            System.out.println("Running KNN algorithm...");
            KNN_Algorithm knn = new KNN_Algorithm();
            try {
                knn.run();
            } catch (FileNotFoundException File_Error) {
                System.out.println("Error when reading KNN test or train file!");
            }
        }
        
        // This will run the Random Guess Number Algorithm
        if (Random_Guess_Algorithm) {
            System.out.println("Running Random Guess algorithm...");
            Random_Guess_Algorithm randomGuess = new Random_Guess_Algorithm();
            try {
                randomGuess.run();
            } catch (FileNotFoundException File_Error) {
                System.out.println("Error when reading Random Guess test or train file!");
            }
        }
        
        // This will run the Self-Organizing Map Algorithm
        if (SOM_Algorithm) {
            System.out.println("Running SOM algorithm...");
            try {
                // Create an instance of SOM_Algorithm and run it
                SOM_Algorithm som = new SOM_Algorithm(10, 10, 64); // Grid size 10x10, input dimensions 64
                som.run();
            } catch (FileNotFoundException File_Error) {
                System.out.println("Error when reading SOM test or train file: " + File_Error.getMessage());
            }
        }
        
        // This will run the C4.5 Decision Tree Algorithm
        if (C45_Algorithm) {
            System.out.println("Running C4.5 algorithm...");
            try {
                // Create an instance of C45 Algorithm and run it
                C45_Algorithm c45 = new C45_Algorithm();
                c45.run();
            } catch (Exception e) {
                System.out.println("An error occurred: " + e.getMessage());
            }
        }
    }
}