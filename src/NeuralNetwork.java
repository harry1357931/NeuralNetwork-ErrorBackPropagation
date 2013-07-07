/* Class NeuralNetwork
 * This Class have all functions that are required to build a Neural Network
 * like initially read file, then randomization of data set, then Actual Network function, 
 * Five Fold Cross validation function
 * Machine Learning - CS 3813
 * **********
 * Parameters
 * **********
 * @param MainDataSet  Contains the actual data points
 * @param NormMainDataSet Contains the Normalize data points
 * @param RandomizedNormMainDataSet Data Points in this Array are just Random Distribution from NormMainDataSet
 * @param WeightsInpHidden  Contain Weights for First Layer that goes from Input to Hidden units W_ij
 * @param WeightsHiddenOut  Contains Weights for Second Layer that goes from Hidden to Final Outputs W_jk
 * @param LR Learning Rate
 * @param Threshold_Error  Threshold Error Value
 * @param CountEpoch Count Number of Epochs
 * @param T_OutputError Total Output Error
 * @param TrainingError Array to Store Training Error for 5 folds
 * @param TestingError Array to Store Testing Error for 5 folds
 * @param TotalOutputError Array to Store Total out put error for five folds
 * @param AvgFiveFoldTotalOutputError  Average Total Output error of five folds
 * 
 * @author Gurpreet Singh 
 */

import javax.swing.JOptionPane;
import java.text.DecimalFormat;
import java.util.StringTokenizer;

public class NeuralNetwork {
    public StringTokenizer myTokens;
    public int TotalDataPoints = CountTotalDataPointsInFile("machine.data.txt");    // String is Text File Containing DataPoints
    
    // Data
    public double[][] MainDataSet = new double[TotalDataPoints][8];          // Main Data Set
    public double[] MaxValues = {1500,32000,64000,256,52,176,1150,1238};
    
    // Normalized Input Data Sets
    public double[][] NormMainDataSet = new double[TotalDataPoints][8];      // Normalized Main Data Set   
    public double[][] RandomizedNormMainDataSet = new double[TotalDataPoints][8];       // This array will be divide into 5 folds...
    public int[] RandomFillings = new int[RandomizedNormMainDataSet.length];
    
    // Weights
    public double[][] WeightsInpHidden = new double [3][7];           // W_ij
    public double[] WeightsHiddenOut = new double [3];                // W_jk
    
    // 5 Fold Cross Validation...
    public int[][] CrossFold = {{0,41,42},{42,83,42},{84,125,42},{126,167,42},{168,208,41}};
    
    // Outputs
    public double[] OutputAtHidden= new double [3];                  // O_j For only One Data Point..
    public double[] InputAtHidden= new double [3];                   // For only One Data Point..
    public double FinalInput = 0, FinalOutput = 0;                   // O_k Predicted Output...for one Data point only...
    
    // Learning Rate
    public double LR=0.1;
    
    // Error
    public double[] Threshold_Error = {0.1, 0.01, 0.0016, 0.00155};
    public double DataPointError=0;
    public double MultipleDatapointsAvgError=1;                      // Average error over all data points for 1 epoch
    public double T_OutputError = 1;
    public int[] CountEpoch = {0,0,0,0,0};
    public double[] TrainingError = new double[5];
    public double[] TestingError = new double[5];
    public double[] TotalOutputError = new double[5];
    public double AvgFiveFoldTrainingError = 0;
    public double AvgFiveFoldTestingError = 0;
    public double AvgFiveFoldTotalOutputError = 0;
    
    /*  Function: Network
     *  Description: 
     *  1) It builds and Train network using TrainingSet
     *  2) Initializes all weights for W_jk and W_ij layer 
     *  3) Using Feed Forward...Compute Final Output
     *  4) Recompute Weights for both layers
     *  5) Check for Threshold termination Condition
     */
    public int Network(double[][] TrainingSet, double Threshold){
    	
    	int EpochCount=0;
    	T_OutputError =1;
    	
    	InitializeWeights();
    	
    	while(T_OutputError > Threshold){                       //Checks Threshold Error
    	  
    	   for(int m=0; m < TrainingSet.length; m++){           // To Loop through All data Points...	
    		   
    		   for(int k=0; k<3;k++){	                        // For Each Data point....this for loop contains the Processing for each data point                    
		    	 for(int i=0; i<3;i++){
		    		for(int j=0; j<7; j++){
		    			InputAtHidden[i] = InputAtHidden[i]+TrainingSet[m][j]*WeightsInpHidden[i][j];
		    		}                                           // j  To get one hidden Output	
		    		OutputAtHidden[i] = Sigmoid(InputAtHidden[i]);  
		    	  }                                             // i   To get 3 hidden outputs
		    	  FinalInput = FinalInput + OutputAtHidden[k]*WeightsHiddenOut[k];    
		        }	                                            // k To get Predicted Output for m-th data point
		      
		       FinalOutput = Sigmoid(FinalInput);               // Predicted Output for m-th Data Point...
		     
		       //Computes Squared Error for m-th Data Point...
		       DataPointError = DataPointError + Get_Error(TrainingSet[m][7], FinalOutput);      
		     		      
		       // Re Compute All Weights using Back Error Propagation                                
		       for(int p=0; p<3; p++){                          // To Re Compute first 3 Weights...Hidden-->Output
		           WeightsHiddenOut[p] = WeightsHiddenOut[p] + LR*(TrainingSet[m][7]- FinalOutput)*FinalOutput*(1-FinalOutput)*OutputAtHidden[p];
		           for(int n=0; n<7;n++){                       // To Re Compute Input-->Hidden Weights...
		            	WeightsInpHidden[p][n] = WeightsInpHidden[p][n] + (LR*(TrainingSet[m][7]- FinalOutput)*FinalOutput*(1-FinalOutput)*WeightsHiddenOut[p]*(OutputAtHidden[p]*(1-OutputAtHidden[p]))* TrainingSet[m][n]);                                                    
		           }
		       }
		     
		       // Resetting Values
		       FinalInput=0;
		       for(int v=0; v < 3; v++){ 
		           InputAtHidden[v] = 0 ;                           // Resetting to Zero for Next Data Point...
		       }
		  }                                                    // Loop Through ALL Data Points Ends here...
    	  EpochCount++;  
    	 
    	 T_OutputError = ComputeNetworkErrorOver(RandomizedNormMainDataSet);
    	 System.out.println("Wait!....LR: "+LR+"   Total Output Error: "+EpochCount+ " : "+T_OutputError);     // One Iteration over all datapoints
    	 DataPointError =0;                                          // Resetting error to zero...for new iteration over 209 data points
    	
       }  // While Average Error greater than threshold Error.
    	
       return EpochCount;
    
    } // Function Network ends here..
    
    /* Function: FiveFold_CV_Training_TestingAndResults
     * Description:
     *  1) Controls Everything from 5 Fold to Training a Network, Testing a Network and Out putting Final reports
     *  2) Call Other Functions to achieve point 1)
     */
    public void FiveFold_CV_Training_TestingAndResults() {      // 5 Fold Cross Validation + Call Network function to Train data
      
      for(int m=0;m< Threshold_Error.length; m++){	          // For Three different Threshold Values 0.1, 0.01, 0.0015
    	  for(int Fold=0; Fold<=4; Fold++){
    	  	double[][] Testing = new double [CrossFold[Fold][2]][8];
    		double[][] Training = new double [209 - CrossFold[Fold][2]][8];
   		    
   		    int TrainingCount=0, TestingCount=0;
     		for(int i=0; i< 209; i++){           // Filling Training and Testing Arrays...for Some fold...
     			
     			if(i>=CrossFold[Fold][0] && i<= CrossFold[Fold][1]){
     				Testing[TestingCount++] = RandomizedNormMainDataSet[i];
     			}
     			else{
     				Training[TrainingCount++] = RandomizedNormMainDataSet[i];
     			}
     		}// i  for loop ends here
     		
      		CountEpoch[Fold] = Network(Training, Threshold_Error[m]);                // Creating a Network Over Training Set...
       		TrainingError[Fold] = ComputeNetworkErrorOver(Training);  
    		TestingError[Fold] = ComputeNetworkErrorOver(Testing);
    		TotalOutputError[Fold] = ComputeNetworkErrorOver(RandomizedNormMainDataSet);
    		System.out.println("\n");
    		
    		AvgFiveFoldTrainingError = AvgFiveFoldTrainingError + TrainingError[Fold]; 
       	    AvgFiveFoldTestingError = AvgFiveFoldTestingError + TestingError[Fold];
     	    AvgFiveFoldTotalOutputError = AvgFiveFoldTotalOutputError + TotalOutputError[Fold];  
         
    	} // Section   for loop ends here....
    	
   	    AvgFiveFoldTrainingError = AvgFiveFoldTrainingError/5;
	    AvgFiveFoldTestingError = AvgFiveFoldTestingError/5;
	    AvgFiveFoldTotalOutputError = AvgFiveFoldTotalOutputError/5;
        OutputFinalReport(m);
	    // Resetting Values to Zeros after Out putting...
	    AvgFiveFoldTrainingError = 0;
	    AvgFiveFoldTestingError = 0;
	    AvgFiveFoldTotalOutputError = 0;
	    
      }//m    For 4 different threshold values....
      
    }// Function Ends here...
    
    public void OutputFinalReport(int m){                      // Output Final Report
     	    	
    	StringBuilder builder= new StringBuilder();
    	DecimalFormat df = new DecimalFormat("#.#######");
    
    	builder.append("Multilayer Neural Network Project 2 by: Gurpreet Singh\n\n");
    	
    	builder.append("Threshold Error Value: "+ Threshold_Error[m]+"         Learning Rate: "+LR+"\n\n");
		builder.append("Average Five Fold Total Output Error: "+df.format(AvgFiveFoldTotalOutputError)+"\n");
		builder.append("Average Five Fold Training Error: "+df.format(AvgFiveFoldTrainingError)+"\n");
		builder.append("Average Five Fold Testing Error: "+df.format(AvgFiveFoldTestingError)+"\n\n");
		
		for(int i=0; i<5; i++){
		
		  builder.append("Fold "+i+":-   Total Output Error: "+df.format(TotalOutputError[i])+"       Epochs(during Training): "+CountEpoch[i]+"\n");
		  builder.append("                Training Error: "+df.format(TrainingError[i])+"\n");
		  builder.append("                Testing Error: "+df.format(TestingError[i])+"\n\n");
		
		}
		
        builder.append("Click on OK Button below to Change Threshold Value to one of 0.1, 0.01, 0.0016 and 0.00155 and then Wait for few Seconds...:-\n");
	    builder.append("\n\nTo Run the Program with different Learning Rate:-\n");
		builder.append("Change the Value of Static Variable 'LR' in Class 'NeuralNetwork' and then Run the Program Again...");
     
    	JOptionPane.showMessageDialog(null, builder.toString());
	
    }  // Function Output Final Reports Ends here

    public double ComputeNetworkErrorOver(double[][] ToFindErrorOn){
    	
	   	DataPointError =0;
        for(int m=0; m < ToFindErrorOn.length; m++){    // To Loop through All data Points...	
   		   // For Each Data point
   		   for(int k=0; k<3;k++){	                                
		       for(int i=0; i<3;i++){
		    	  for(int j=0; j<7; j++){
		    			InputAtHidden[i] = InputAtHidden[i]+ToFindErrorOn[m][j]*WeightsInpHidden[i][j];
		    	  }                                  // j  To get one hidden Output	
		    	  OutputAtHidden[i] = Sigmoid(InputAtHidden[i]);
		    	}                                    // i   To get 3 hidden outputs
		    	FinalInput = FinalInput + OutputAtHidden[k]*WeightsHiddenOut[k];    
		    }	                                           // k To get Predicted Output for m-th data point
		      
		    FinalOutput = Sigmoid(FinalInput);              // Predicted Output for m-th Data Point...
		     
		    //Computes Squared Error for m-th Data Point...
		    DataPointError = DataPointError + Get_Error(ToFindErrorOn[m][7], FinalOutput);      
		     		       
		    // Resetting Values
		    FinalInput=0;
		    for(int v=0; v < 3; v++){ 
		           InputAtHidden[v] = 0 ;                           // Resetting to Zero for Next Data Point...
		    }
		    
		 }     // Loop Through ALL Data Points
   	 
   	    MultipleDatapointsAvgError = DataPointError/ToFindErrorOn.length;   
   	    return MultipleDatapointsAvgError;

    }// Function Ends here...
    
    
    public static double Get_Error(double Target_Output, double Predicted_Output){     // Return Squared Error
    	 	return (0.5)*(Math.pow(Target_Output - Predicted_Output,2));
    }
    
    public static double Sigmoid(double input){                           // Sigmoid Function
       double output=0;
       output = 1/(1+Math.exp(-1*input));
       return output;
    }
    
    public void InitializeWeights(){
         
    	for(int i=0; i<3;i++){
    	  for(int j=0; j<7;j++){
    		 WeightsInpHidden[i][j] = Math.random()*0.01;      
    	  }	
    		WeightsHiddenOut[i] = Math.random()*0.01;         
    	}
    } 
    
    public double[][] readFile(String fileName, int arraylength, int SubArrayLength)        // Length of array to be filled  
	{   
	    double[][] loaded = new double[arraylength][SubArrayLength];
	    
	    TextFileInput tfi = new TextFileInput(fileName);
		String line = tfi.readLine(); 
	    int count=0;
		while(line!=null)
		{	myTokens = new StringTokenizer(line," ,");
		    int j=0;
		    myTokens.nextToken();                // To Remove First Extra Name
		    myTokens.nextToken();                // To Remove Second Extra Name
			while(myTokens.hasMoreTokens()){
			   loaded[count][j] = 	Integer.parseInt(myTokens.nextToken());
			   j++;
			}
			count++;      
			line=tfi.readLine();
			
		}// while loop ends here
		
		return loaded;           // "loaded" array is an array formed by reading from file...
  	}// Read File method ends here
	
	public double[][] FillNormalizedValues(int arraylength, int SubArrayLength){
		
		double[][] Normalized = new double[arraylength][SubArrayLength];
		
		for(int i=0; i< arraylength; i++){
		   for(int j=0; j< SubArrayLength; j++){
			  Normalized[i][j] = MainDataSet[i][j]/MaxValues[j];  	
		   }
        }
	    
		return Normalized; 
    }  
	
	 public static int CountTotalDataPointsInFile(String fileName)        // length of array to be filled  
     {   
		    int CountLine=0;
		    TextFileInput tfi = new TextFileInput(fileName);
			String line = tfi.readLine(); 
		    while(line!=null)
			{	line=tfi.readLine();
				CountLine++;
			}
		    
			return CountLine;                                         
	 }//  Method ends here

	 public static void print_array( double[][] ToPrint, int SubArrayLength ){		
			
			for(int i=0; i< ToPrint.length; i++){
				for(int j=0; j < SubArrayLength; j++){	
		           System.out.print(ToPrint[i][j] + ",");			
				}
				System.out.println();
		    }
	 } // print_array ends here
	 
	 public void FillRandomizedNormMainDataSetArray(){
		 
		 int aNumber;
		 for(int i=0; i< RandomFillings.length; i++){            // Selecting Random Numbers
			 while(true){
	    		  aNumber = (int) (Math.random() * (209) + 0);     
	    		  if(NumExistsInArray(aNumber, i)== false)
	    			  break;
	    	 }
			 RandomFillings[i] = aNumber;
		 }
		 
		 for(int i=0; i<RandomizedNormMainDataSet.length;i++){  // Filling RandomizedNormMainDataSet array from NormMainDataSet Randomly...
			 RandomizedNormMainDataSet[i] = NormMainDataSet[RandomFillings[i]];
		 }
		 
	 }// function ends here...

	 public boolean NumExistsInArray(int NumToBeChecked, int UpToIndex){
	    	if(UpToIndex == 0)
	    		return false;
	    	
	    	for(int i=0; i< UpToIndex; i++){
	    		if(RandomFillings[i]== NumToBeChecked){
	    			return true;
	    		}
	    	}
	    	
	    	return false;
	  }// Function Ends here...
			
}// Class Neural Network Ends here...
