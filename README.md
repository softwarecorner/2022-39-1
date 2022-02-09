# Deep learning with Keras in R

Stephanie Clark

School of Mathematics and Physical Sciences,
University of Technology Sydney

_Editor note: This article was originally published in the [Biometric Bulletin (2022) Volume 39 Issue 1](https://www.biometricsociety.org/publications/biometric-bulletin). The example code is included in `examples.R`._

There’s a lot of talk these days about big data and the power and benefits of deep learning algorithms. The increasing ability to collect and disseminate all types of data, often in real-time, is generating much larger data sets than have historically been available in many fields, leading many statistical researchers to speculate whether machine learning algorithms such as deep learning networks might be of value for efficiently analysing their particular set of data.

However, the perceived leap from traditional statistical modelling in R into the more computer-science-based world of Python, where much of the development of deep learning techniques occurs, may be enough to inhibit the exploration of these possibilities for some statisticians. Granting that R may be the most comfortable option for many researchers in terms of pre-processing the data and visualising the results, transferring the middle portion of an analysis into and out of Python is not a very appealing prospect. Fortunately, a bridge exists in the form of Keras for R, an easy to implement, flexible connection from R to the backend of TensorFlow which performs the deep learning analyses. Keras in R opens the possibility of containing the entire project within a single R workflow without switching the analysis entirely or partially into Python to access the function of TensorFlow.

## A brief background on neural networks and deep learning

In general, while classical statistical analyses are focussed on inference and providing detailed explanations of a system, machine learning places more importance on efficiently and accurately making predictions of system behaviour. Neural networks can be used for classification or regression tasks, and are particularly beneficial when working with large data sets, where the underlying system is not well understood (possibly containing nonlinear relationships between components), where the predictor space is not well defined (the neural network does the feature engineering), when working with diverse data sources, and for analyses in real time (it can be trained on existing data and updated as new data comes in). Common applications are in image pattern recognition (facial or health images), meteorology, ecology, time series pattern analysis and prediction, anomaly detection, economics and natural language processing - though the creativity of researchers is continually expanding the application of these techniques to new areas. 

The most basic neural network, known as a multi-layer perceptron (MLP), comprises a set of interconnected nodes arranged into layers - an input layer, one or more hidden layers, and an output layer. It is through the inclusion of the hidden nodes that the MLP can discover important interactions between the input variables needed for constructing a good prediction of the outcome. Each connection between nodes has an associated weight (statisticians would tend to use the word parameter instead) and as data flows between each layer a bias is added to the summed weighted inputs before the data is run through an activation function at each node. The activation function allows for a nonlinear mapping between the inputs and outputs of each node. An example of a 2-layer MLP is shown below. Recurrent neural networks (RNNs) are an extension of the MLP for time series data - the input data are no longer processed in a single step, instead the model loops through each time step sequentially with a ‘state’ being retained and updated in the hidden layer. Training a neural network is the process of fitting the model and estimating the unknown model parameters (weights and biases). This is an iterative process of feeding observed data through the model, comparing the model prediction to the observed output, and incrementally improving the parameters until the output best resembles the measured data.

![Untitled](https://user-images.githubusercontent.com/2189134/153106722-332206bc-85dd-4800-9779-bef9b47cab3f.png)

Deep learning extends traditional neural networks to include a multitude of layers, each characterised as a potentially nonlinear function of the output from the previous layer. Both the MLP and RNN have deep learning counterparts: the convolution neural network (CNN) for multi-layer analysis of 2D data (ie. images) and the long short-term memory algorithm (LSTM) for multi-timestep analysis of time series data. In CNNs, which make up the main portion of deep learning applications, each output is a function of a small number of neighbouring inputs. These networks can have up to hundreds of thousands of layers, with each layer performing a transformation to the data before passing it to the next layer. In LSTMs, which model sequential data such as long time series, each output is a function of all previous members of the input, with important information retained over time in memory cells and unimportant information forgotten. Deep learning algorithms are especially useful for analysing data with complex spatial or sequence dependencies, and data that require a lot of feature engineering. For more information on deep learning in general, see Goodfellow et al (2016) which is freely available online.

## What is Keras?

Keras is an API (application programming interface) designed for deep neural networks and built in Python. Keras allows easy interaction between the user and the backend application that will ultimately perform the deep learning computations, through an intuitive facilitation of model setup and data input/retrieval. TensorFlow (developed by Google) is the default backend for Keras, providing a platform for machine learning computations. Keras also runs with other backends such as Theano (University of Montreal) and CNTK (Cognitive Toolkit, Microsoft). Other Keras resources include built-in datasets, data set generators, and pre-trained models that can be used for feature extraction, fine-tuning and prediction with new data. For more information on Keras in Python, visit https://keras.io/getting_started/. 

In R, the ‘keras’ package (Chollet & Allaire, 2017) is an interface to the Python Keras package, allowing the user to remain working in R while accessing TensorFlow to build and run their deep learning model. It enables the running of CNNs, LSTMs and various combinations of these. Two books (with the same name!) provide very useful information ranging from the basics of neural networks up to advanced deep learning implementations in R with Keras (Chollet & Allaire, 2018; Ghatak, 2019).

## Getting started

Keras in R is installed through the usual `install.packages()` command followed by a further `install_keras()` command, as in:

```r
install.packages(“keras”)
library(keras)
install_keras()
```

A **tensorflow** package also needs installation in the same way. As Keras in R is an interface to Keras in Python, it is necessary to have Python installed also.

Once installed, the use of Keras in R is straightforward. Data is preprocessed, the model architecture is set up, hyperparameters are chosen, training is performed, the model is evaluated, and finally the model can be used for predictions. 

## Data setup

As TensorFlow requires data in a tensor format, this usually requires some rearranging of a set of measurement data. Keras provides data set generators which can do this for you, either for use in CNNs or LSTMs. For CNNs, an image data generator automatically converts image files into pre-processed tensors. A jpeg is automatically decoded into an RGB grid of pixels, rescaled and set into batches. For sequential data, a time series data generator automatically converts temporal data into sequential batches of input/output.

Input data is usually normalised by variable and categorical variables need to be integer-encoded. The data set is then split, usually into three portions: training data, used to train the model; validation data, used to mitigate overfitting during the training process; and testing data, a portion that is withheld from the training process and is used to test the generalisation ability of the trained model. The ratio of this data split is often of the order: 60%, 20%, 20%. For time series data, the split is sequential, but for tabular data it is usually random.

## Model setup

The layout of the network, such as the number of layers and the number of nodes on each layer, is set through Keras. Generally the multiple layers will be set with a decreasing number of nodes on each. This is done through the `keras_model_sequential()` command, which is as easy as adding an additional line for each additional network layer, as shown below for a 2-layer network with 64 nodes on the first hidden layer and 32 on the second. Here there is a single output node indicating that this model is either for regression or binary classification. If it were for an n-class classification task, the final layer would instead have n units.

```r
model = keras_model_sequential() %>%
  		      layer_dense(units=64,  activation="relu", input_shape=ncol(x_train)) %>%
  		      layer_dense(units=32, activation="relu",) %>%
    	    	layer_dense(units=1, activation="linear")
```

Because neural networks tend to be very highly parameterised, it is often helpful to use some kind of regularisation to stabilize the model fitting process. Keras provides several options, including dropout or weight decay regularisation. Dropout involves the random deactivation of a small percentage of the nodes in each hidden layer during the training phase, preventing overfitting by lessening the dependence of the model on individual nodes and providing a more balanced representation of the data by the model. Weight decay, or L2, regularisation prevents overfitting through the penalisation of large weights, shrinking them according to a pre-set regularisation hyperparameter. Either regularisation method can be added during specification of the model architecture. For example, the above model with 20% dropout on each hidden layer would be set as: 

```r
model = keras_model_sequential() %>%
          layer_dense(units=64,  activation="relu", input_shape=ncol(x_train)) %>%
    	    layer_dropout(dropout=0.2) %>%
  	      layer_dense(units=32, activation="relu") %>%
    	    layer_dropout(dropout=0.2) %>%
    	    layer_dense(units=1, activation="linear")
```

## Training

Before beginning training, the loss function and updating mechanism, or optimiser, must be specified. The loss, or difference between the measured and predicted values, is used as feedback to an optimisation algorithm that guides the updating of the weights and biases. For regression this is often MSE, and cross-entropy loss for classification. Optimizers, such as the stochastic gradient descent algorithm or Adam, aim to diminish the error with each step, updating weights so that the error will be smaller on the next iteration. There are a number of choices for loss and optimisation which are outlined on the Keras webpage. Simple code is used for compiling the model with its relevant loss function and optimisation choices:

```r
model %>% compile(loss = "mse", optimizer =  "adam", metrics = list("mean_absolute_error"))
```

Now the model is ready for training, which involves an iterative updating of the weights and biases until the model output matches the observed response data in the training set. The number of training epochs, batch size, data to be used for the training predictor and response variables (`x_train`, `y_train`), and validation data (`x_val`, `y_val`) are set within the `fit()` function. To reduce memory requirements and speed up training, data is input to the model in batches (or subgroups of observations) with an update of model parameters after each batch. One epoch of training is complete when the model has seen all batches that make up the dataset. 

```r
history <- model %>% fit(x_train, y_train,
   		      epochs = 200, batch_size = 32,
  		      validation_data = list(x_val, y_val))
```

In practice, it will be important to repeat the analysis over a broad range of choices of the various hyperparameters (eg. number of layers, number of nodes per layer, regularisation method, etc.) and choose the hyperparameters for use in the final model that produce the best results.

## Evaluation and prediction

To test the generalisability of the ultimately chosen model, the trained model is evaluated on data not seen by the model during the training process. The root mean squared error of a regression estimate or the accuracy of classifications can be determined, providing a measure of how effective the model is at making predictions on previously unseen data.

```r
model %>% evaluate(x_test, y_test)
```

Finally, predictions can be made with the trained model. A list of predicted y-values or classes for each item of `x_test` can be found with:

```r
model %>% predict(x_test)
model %>% predict_classes(x_test)
```

## Application

We recently used Keras to model time series corresponding to daily measurements of groundwater levels as a function of rain and other climate variables. As a simple example, the following figure shows the observed groundwater levels in blue, along with rainfall measurements in turquoise. The split of the data into training, validation and testing segments can be seen. The orange line shown in the testing segment represents the predictions constructed by an LSTM using the training and validation components of the dataset. See Clark _et al_ (2020) for more discussion. 

![Untitled1](https://user-images.githubusercontent.com/2189134/153107580-120ff3e3-e2b1-47c3-9506-4853e047c2c7.png)

## References and additional resources

- Chollet, F. & Allaire, J. J. (2017). _R interface to keras._ https://github.com/rstudio/keras
- Chollet, F. & Allaire, J. J. (2018). _Deep learning With R._ Manning Publications Company.
- Clark, S., Hyndman, R.J., Pagendam, D. and Ryan, L.M. (2020). _[Modern strategies for time series regression.](https://doi.org/10.1111/insr.12432)_ International Statistical Review, 88, pp.S179-S204. 
- Ghatak, A. (2019). _Deep Learning with R._ Springer.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). _Deep Learning._ MIT Press. http://www.deeplearningbook.org

The following webpages provide a wealth of explanations, guidance and examples of deep learning in R:

- Keras in R: https://keras.rstudio.com/index.html
- Examples of applications: https://blogs.rstudio.com/ai/gallery.html
- More basic guidance: https://tensorflow.rstudio.com/guide/
