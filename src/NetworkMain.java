/* Class NetworkMain
 * Create and Initiate a Neural Network Object 
 * Machine Learning - CS 3813
 * @param MLNeuralNetwork  Network Object
 * @author Gurpreet Singh 
 */

public class NetworkMain {
    public static NeuralNetwork MLNeuralNetwork = new NeuralNetwork();             // Creates a NeuralNetwork Object
	
    /* Main Method
     * read file "machine.data.txt" and store it in 2D array
     * Randomize Data set and Make it ready for 5 fold Cross Validation Process
     * Calls Method "FiveFold_CV_Training_TestingAndResults()" which controls Everything  
     */
    public static void main(String[] args) {
		
		MLNeuralNetwork.MainDataSet = MLNeuralNetwork.readFile("machine.data.txt", MLNeuralNetwork.MainDataSet.length, 8);
		MLNeuralNetwork.NormMainDataSet = MLNeuralNetwork.FillNormalizedValues(MLNeuralNetwork.MainDataSet.length, 8);
		MLNeuralNetwork.FillRandomizedNormMainDataSetArray();           // Randomize the Data Set
	    // This Function is Main Controller...
		MLNeuralNetwork.FiveFold_CV_Training_TestingAndResults();       // 5 Fold Cross Validation Function
	}
		
}// Class NetworkMain
