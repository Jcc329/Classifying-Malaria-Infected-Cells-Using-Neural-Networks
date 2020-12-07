# Classifying Malaria Infected Cells Using Neural Networks
Final Project for Data 602, Machine Learning
By <a href="https://github.com/Jcc329">Jessica Conroy Styer</a>

## Table of Contents
<b> Notebooks may run faster or slower based on the computational resources available </b>
<ul>
  <li><b>Notebooks</b> - The Jupyter notebooks used to complete this project.</li>
  <ul>
    <li><a href="https://github.com/Jcc329/Classifying-Malaria-Infected-Cells-Using-Neural-Networks/blob/main/Jupyter%20Notebooks/Data%20Cleaning%20_%20Final%20Project.ipynb">1. Data Cleaning</a></li> - The notebook showing how the initial data was acquired and processed to create the cleaned dataset.
    <li><a href="https://github.com/Jcc329/Classifying-Malaria-Infected-Cells-Using-Neural-Networks/blob/main/Jupyter%20Notebooks/Exploratory%20Analysis_Final%20Project.ipynb">2. Exploratory Analysis</a> - The jupyter notebook containing exploratory analyses (takes much longer to run) </li>
    <li><a href="https://colab.research.google.com/drive/11Me6TJWDqCsK-7zHNAliLxNJS2_HFwJU?usp=sharing">2.5 Google Colab Exploratory Analysis</a> - The link to the Google Colab notebook containing exploratory analyses and the tensorboard comparison, recommended for faster execution. Note: This is not identical to the jupyter notebook as it replaces the manual checking of parameters with a tensorboard instance. </li>
    <li><a href="https://github.com/Jcc329/Classifying-Malaria-Infected-Cells-Using-Neural-Networks/blob/main/Jupyter%20Notebooks/Technical%20Report_Final%20Project.ipynb">3. Technical Report</a> - The technical notebook containing the report and final model used. </li>
  <li><b>Image Data</b> - Example parasitized and uninfected images. The notebooks include data aquisition programatically</li>
  <li><a href="https://github.com/Jcc329/Classifying-Malaria-Infected-Cells-Using-Neural-Networks/blob/main/README.md">README.md</a> - An overview of the project and results</li> 
</ul>

### Overview

Malaria is a parasitic disease which can be fatal if not treated propperly. Transmitted via a mosquito vector, malaria disproportionately impacts poor, subtropical areas and is the leading cause of death in many developing countries (Ahmad & Ahmad, 2020). As a result, any intervention that may improve detection and treatment of Malaria with minimal cost, would have a positive impact on at risk populations in these areas. Currently, the primary methods of diagnosing Malaria is based upon detection of antibiodies in the serum of the blood using microscopy, serology, or a rapid diagnostic test. There are several drawbacks to these methods, microscopy and serology are time and effort intensive, as well as limited by the quality of those performing the test. And while RDTs are much faster, they are less accurate than the other two methods (CDC.gov, 2020; Wilson, 2013). Using a dataset of labeled cell images, some with the malaria infection, and some healthy cells, I aimed to train a neural network model that would detect the infected cells with a high degree of accuracy and precision. Accuracy of laboratory testing for Malaria varies, but a recent study found an accuracy from a Belgium lab to be around 97%-98%. It was therefore my goal to obtain at least an accuracy of 97%, in other words, successfully labelling 97% of positive cases as positive, with my model. 

By creating a model that can detect the infection from a photo, I hope to reduce the amount of human expertise required to diagnose a patient, which often means that patients need to wait while samples are sent to a laboratory where staff manually examine them, and instead replace it with a much faster process, whereby a photo of the sample, taken with a microscope on premises, can be run through the model and a determination quickly produced. As a result, providing faster results and greater capacity in the pipeline. 

Using a neural network model, I tested a range of hyperparameters (padding, activation function, input units, number of layers, and optimizers) to train a model with the highest possible accuracy. The resulting model had valid padding, a relu activation function, an additional data augmentation layer (to prevent overfitting), and four 2D convolutional layers, along with four maxpooling layers. The final model also contained a dropout layer with a 20% rate to minimize overfitting and used SGD as the optimizer. 

The resulting validation accuracy on the model was between 96-97%, very close to my target. My test accuracy was slightly lower, at 95.2%.

### Research Questions:

The question to be addressed in this process is as follows:

1. Using only a photo of a blood cell, can I predict with a level of acccuracy equivalent to or higher than microscopy in a lab (97%), whether a Malaria infection is present? 

### Motivation & Background

There were several factors that led to my decision to use this image data. First, was the specific desire to work with image data and train a model using image data because that is a skill I have not used in the past. A general interest in population health data led me on a search for data in that field. I ultimately decided on the Malaria image dataset because the data were new, in that they had been posted only recently, and no one else had performed analyses or posted any models yet. The novelty meant that I could be among the first people to examine and potentially produce a model using these data, which was appealing. 

### Data

The data used were collected from Kaggle: https://www.kaggle.com/syedamirraza/malaria-cell-image?select=cell_images
They consist of labeled images of malaria infected and healthy cells. 

<img src="Image Data/C100P61ThinF_IMG_20150918_144104_cell_162.png" alt="Parasitized cell"> 
<img src="Image Data/C1_thinF_IMG_20150604_104722_cell_15.png" alt="Uninfected cell">
Left: Parasitized, Right: Uninfected

### Conclusion

While I was not able to train the model to a target accuracy of 97%, I was able to get to a test accuracy of about 95.2%. Despite being lower than the accuracy produced by lab studies, using a model like this would still provide some benefits. For example, running an image into the model and receiving a prediction would still occur in a faster time frame than the time needed to send a sample to a lab to be processed. In addition, the accuracy is far better than the current rapid tests, which means that this could be a potential replacement for those tests, saving time and therefore money in the diagnosis process of malarial treatment. 

Through this project I was able to gain a lot of experience using image data in python as well as using the tensorflow and keras libraries. I feel like I have gained a firmer grasp of the architectural components of neural networks, how they change, and what the benefits of different variations are. This has been an extremely interesting, not to mention fun, project and I feel like I was able to use and gain experience with a lot of novel concepts and  their applications.

Future reasearch could be aimed at expanding the methodologies used here to train the model on a range of blood infections, thus allowing the model to identify several different possible diagnoses, rather than just Malaria, from a single sample. As a result, a single image of a sample could potentially be used to rule out, or rule in, several different diagnoses at once, a vast improvement on many of the models currently in existence. 

### Packages and Software

Software:
Python 3.0
Anaconda 3
Jupyter Notebooks
Google Colab Pro

Packages:
kaggle
matplotlib
numpy
os
PIL
tensorflow
pathlib
keras

### References and Contributions

Ahmad, A., & Ahmad, S. (2020). Taming the beast: Update on Malaria Research. molecules, 7(07).

“CDC - Malaria - Diagnostic Tools.” Centers for Disease Control and Prevention, Centers for Disease Control and Prevention, 19 Feb. 2020, www.cdc.gov/malaria/diagnosis_treatment/diagnostic_tools.html. 

Wilson, M. L. (2013). Laboratory diagnosis of malaria: conventional and rapid diagnostic methods. Archives of Pathology and Laboratory Medicine, 137(6), 805-811.

Blogs, Tutorials, and webpages:
https://www.tensorflow.org/tutorials/images/classification

https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363

https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks
