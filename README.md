# Dog Breed Identification – Deep Learning Project
## Project Description
This project intends to correctly recognize the breed of dogs in an image. The motivation behind this project is to get an idea of what breed my dog could belong to. His name is Kimi and he is suspected that he is a cross breed dog that looks a bit like a Beagle but is bigger and taller.
This application can be useful to help recognize people the different breeds of dog that exist.
For the development of this project, transfer learning has been taken advantage of, the followings pretrained models from Keras applications have been considered: InceptionResNetV2, ResNet50V2, VGG16, InceptionV3.

## Data
The data used is the Stanford Dog Breed dataset collected from Kaggle, with data available for 120 breeds. This data includes between 60 and 120 images for each different breed. The training set consists of a total of 10222 images, the test is not going to be used, validations would be performed in the validation set extracted from the training set and the final test would be trying to predict Kimi’s, my dog, breed. The data can be downloaded here: https://www.kaggle.com/competitions/dog-breed-identification/data

## EDA
To prepare the data for the training process, it was necessary to: check how many training images for each breed were available, create an ImageDataGenerator object to generate augmentation in the training images, split the data between training and validation sets, normalize the pixel values for each RGB color between 0 and 1, dividing the values by 255.
Some sample of the images from the training set were printed with the label of the breed they belong to, that was checked and validated to be confident that there is not wrong data coming in the models.

## Deep Learning
For the development of this project, transfer learning has been taken advantage of, the followings pretrained models from Keras applications have been considered: InceptionResNetV2, ResNet50V2, VGG16, InceptionV3 with the imagenet weights. Each of these models was trained with two optimizers: Adam and RMSprop, removing the top layer and adding a simple pair of Dense layers to get the output needed. The models were trained for 20 epochs and the best performer on this stage was InceptionResNetV2 paired with the Adam optimizer.
After selecting the model, one more time the top layer was removed. However, this time a more complex set of Dense and Dropout layers for regularization was added. After training the model it was possible to get a precision of 70%.


## Findings
After plotting the results obtained of accuracy and loss for the train and validation stages, it was noticed that the model learned fast, almost all the progress was made in the first two epochs and from then onwards it struggled to make meaningful progress.
Looking at a confusion matrix created from a batch of 64 images taken randomly from the validation set, it was noticeable that the model made a good job predicting the breed of most of the images. However, it was detected that it was consistently misclassifying some pair of breeds, those that might looked like one another. The results from the confusion matrix were aligned with what was expected for an accuracy of 70%.
That was the reason for including the second closest breed in the results for each image. This proved to work well. In most cases, when the prediction was mistaken, the second suggestion would get the breed of the dog right.

For our case, it was found that Kimi was classified as: Beagle, English Foxhound and Bluetick. Those three breeds popped in as the first or second option for every test image of Kimi, he might be the result of a combination of those three breeds, most probably a combination between a Beagle and an English Foxhound.

## How can it be better?
* The model could be fine-tuned a bit more, running more epochs and experimenting more with the hyperparameters.

* With a larger dataset, with more individual samples per breed, the model could have gotten even better results.

* A simple yet functional web app could be developed to make the process of using a particular test image way simpler.
