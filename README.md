# A-COMPARATIVE-STUDY-OF-FFN-AND-CNN-WITHIN-IMAGE-RECOGNITION
A Thesis project conducted by Linus Lindahl and Magnus Knutsson at University of Skövde (Sweden)

This repository is the source code used in this thesis project
The folder SourceFiles contains **Training.py** and **results.py** the folder ModelTemplates contains the sourcefiles that creates each individual model. The folder ModelOutputs contains the model performance during training of the models for instance *CNN_Leaky_ReLU_epoch(100)k(10)seed(123).pkl* shows the contains the CNN Leaky ReLU training and the results for each model is presentend in the \<model\>(statistics).txt files for each model, and these are further used by Statistics.xlsx. The report for the thesis project is also presented in the repository.
**The actual models used was to large to store in a Github repository (approximately 5Gb) so they can be found on**:
https://drive.google.com/drive/folders/1nlzaW7kCEkgTnWjSFBDsIzCK4GcylPON

## /SourceFiles/Training.py
This file trains one model at a time, the configuration options is listed below:

- input_x = the width of the input image (32 in this case for CIFAR-10)
- input_y = the height of the input image (32 in this case for CIFAR-10)
- color_depth = the color depth of image (3 in this case RGB for CIFAR-10)
- batch_size = the amount of images in the training set you train on before updating the network (32 in this case)
- epochs = the amount of times the model trained on the entire training dataset (100 in this case)

- k_fold_split = the number of parts the dataset it split into and trained and evaluated on (10 in this case)
- model_architecture = the model that should be trained (FFN Leaky ReLU in this case, the available models to train is listed in the enum *model_type* **important** it is important that the file *modelTemplates/FFNLeakyReLuModel.py* because this is the file that creates the model used)

- random_seed = This is used to seed the random values in order to make the results replicable (123 in this case)

## /SourceFiles/Results.py
This file evaluates the results of all 12 models, does two tailed paired t-tests to see if there is any statistical difference between the amount of epochs needed to train and the validation accuracy achieved by the models. These are the 12 models that are used:
*FFN Base, FFN Deep, FFN Wide, FFN Sigmoid, FFN TanH, FFN Leaky ReLU
CNN Base, CNN Deep, CNN Wide, CNN Sigmoid, CNN TanH, CNN Leaky ReLU*

- epoch = the amount of training done by the model (100 in this case, it is **important** that the model is actually trained for this amount of epochs, it will also be used in the file name for the models)
- k = the amount of configurations the dataset was used during training and evaluation (10 in this case)
- seed = the random seed used during training, this will be used in order to find the correct model, since its a part of the file name (123 in this case)

- tKFold = the t value used in the experiment (4.781 in this case, this will however just be printed to the output files)

## Statistics.xlsx
This file was used to visualize the results as graphs and tables
