NeuralNetwork-ErrorBackPropagation
==================================

The program in this repository creates a 3 layer neural network with 1 hidden layer to find to find 
out a non linear function that best fits the input data. In this program, i implemented Error Back 
Propagation Algorithm to train the neural network and used Gradient descent to find out local minimum of 
the function. Then, i used 5 fold cross validation to find out average fitness of a model.

For Algorithms used and Experimental Results:
=============================================
Check File: <b>FinalReport2.doc</b>

MultiLayer Neural Network:
==========================
In machine learning and computational neuroscience, an artificial neural network, often just named a 
neural network, is a mathematical model inspired by biological neural networks. A neural network consists 
of an interconnected group of artificial neurons, and it processes information using a connectionist 
approach to computation. In most cases a neural network is an adaptive system changing its structure 
during a learning phase. Neural networks are used for modeling complex relationships between inputs and 
outputs or to find patterns in data. 
<br>
For a good summary of <b>Artificial Neural Network</b>:<br>
Check File: <b>FinalReport2.doc</b><br>
Visit Link: http://en.wikipedia.org/wiki/Artificial_neural_network

ErrorBackPropagation:
=====================
Backpropagation, an abbreviation for "backward propagation of errors", is a common method of training 
artificial neural networks. From a desired output, the network learns from many inputs, similar to the way 
a child learns to identify a dog from examples of dogs. It is a supervised learning method, and is a 
generalization of the delta rule. It requires a dataset of the desired output for many inputs, making up 
the training set. It is most useful for feed-forward networks (networks that have no feedback, or simply, 
that have no connections that loop). Backpropagation requires that the activation function used by the 
artificial neurons (or "nodes") be differentiable.
<br>
For better understanding, the <b>backpropagation learning algorithm</b> can be divided into two phases: 
propagation and weight update.
<br>
<b>Phase 1: Propagation</b><br>
Each propagation involves the following steps:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1) Forward propagation of a training pattern's input 
through the neural network in order to generate the 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;propagation's output activations.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2) Backward propagation of the propagation's output 
activations through the neural network using the training 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pattern target in order to generate the deltas of all output and hidden neurons.

<b>Phase 2: Weight update</b><br>
For each weight-synapse follow the following steps:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1) Multiply its output delta and input activation to get the gradient of the weight.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2) Bring the weight in the opposite direction of the gradient by subtracting a ratio of it from the weight.<br>
<br>
This ratio influences the speed and quality of learning; it is called the learning rate. The sign of the gradient of a weight indicates where the 
error is increasing, this is why the weight must be updated in the opposite direction.
<br>
Repeat phase 1 and 2 until the performance of the network is satisfactory.<br>
Source: Wikipedia
For More Info. on ErrorBackPropagation: <br>
http://en.wikipedia.org/wiki/Backpropagation

Input File having datapoints:
=============================
'machine.data.text'

Output:
=======
The program will ouput average 5 fold error on Testing sets, number of epochs and Testing error for each fold. <br>
<b>Sample Output Images</b> when Input file is <b>machine.data.text</b> <br>
 &nbsp;&nbsp;&nbsp;&nbsp; Output_at_ThresholdError_0.01_&_LearningRate_0.1.png   <br>
 &nbsp;&nbsp;&nbsp;&nbsp; Output_at_ThresholdError_0.1_&_LearningRate_0.1.png   <br>
 &nbsp;&nbsp;&nbsp;&nbsp; Output_at_ThresholdError_0.0016_&_LearningRate_0.1.png   <br>
 &nbsp;&nbsp;&nbsp;&nbsp; Output_at_ThresholdError_0.00155_&_LearningRate_0.1.png   <br>
