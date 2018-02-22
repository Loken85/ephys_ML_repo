README:

 Code for analysis presented in 'How much does movement and location encoding impact prefrontal cortex activity? 
 An algorithmic decoding approach in freely moving rats.'

 by Adrian Lindsay, Barak Caracheo, Jamie Grewal, Daniel Leibovitz, and Jeremy Seamans

 Author: Adrian Lindsay
 Email: adrianj.lindsay@gmail.com


Links to the reference documentation for the packages and programs used to create and run this code. 

Numpy/Scipy https://docs.scipy.org/doc/

Scikit-learn http://scikit-learn.org/stable/documentation.html

Keras https://keras.io/

Theano http://deeplearning.net/software/theano/

CUDANN http://docs.nvidia.com/deeplearning/sdk/index.html


Quick guide on setup:

	The code provided here was developed and run on Windows 8/10. In order to run this code yourself you will need 
	a distribution of Python 2.7. You will also need to install the numpy, scipy, and scikit-lean packages. The deep
	learning code framework is written using the Keras package, which requires either Theano or Tensorflow as a backend
	to do tensor operations. Currently, if you are developing on Windows, Theano is a much more straightforward option.
	If you are developing on MacOS or Linux, either backend will work. If you would like to run this code with GPU
	acceleration, you will need support software. This code was tested using Nvidia's CUDA NN on a GTX 1080. Note: be
	aware that training even the modestly sized deep learning networks presented here is extremely computationally
	intensive, and will likely take a very long time without GPU acceleration or other parallelization solutions. 


Handy Hints for using this code for encoding analysis:

	All the provided scripts are applications of supervised learning, predicting output factors (of various dimension
	and form) from input matrices of time-binned spike rates for groups of neurons from electrophysiology recordings.
	
	For portability and modifiability, the scripts can easily be separated into three sections. 
		-Data processing: Common across most of the algorithms used, the code to import and process data for use can 
		easily be swapped out to make use of different data sets. Scaling, normalizing/standardizing, and other
		pre-processing steps can have a major impact on the performance of predictive algorithms. Make sure to 
		experiment if you are trying these techniques on something new. Different data splitting techniques may
		also be required for different data sets / problems, and keep in mind that different data splits will affect
		how learning models behave. 
		-Model Definition and training: Even if you use the learning models as provided there are a number of parameters
		that will drastically change how the model performs on a given data set. Again, feel free to experiment if you are
		running these on a different data set. Note that the parameters, and the model architecture in the provided scripts 
		are a product of extensive testing and optmimization on our data, but are by no means the "best". There are 
		trade-offs to be made between model complexity, fit, and reliability. For concrete examples, look in the DL_grid_CNN 
		and DL_grid_RNN files. 
		-Model Evaluation and file output: Again, mostly common across the algorithms used. These could easily be pipelined
		into other scripts or programs to do further analysis if you wish.
		
	Have fun deep learning!

		