# Machine Learning Practice Project


## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
  - [Iris Dataset](#iris-dataset)
  - [ORL Dataset](#orl-dataset)
- [Usage](#usage)
  - [Setup](#setup)
  - [Running Experiments](#running-experiments)
  - [Configuration](#configuration)
- [Experiments](#experiments)
  - [Experiment 1: A naive approach to classification](#experiment-1-a-naive-approach-to-classification)
  - [Experiment 2: Improved regression](#experiment-2-improved-regression)
  - [Experiment 3: Good o' cross entropy](#experiment-3-good-o-cross-entropy)
  - [Experiment 4: PCA + network](#experiment-4-pca--network)
  - [Experiment 5: PCA + LDA](#experiment-5-pca--lda)
- [Results](#results)


## Project Overview
This project is a practice implementation of various machine learning components. It includes a simple framework designed to train neural networks. The project utilizes two datasets: the Iris dataset and the ORL (Olivetti Faces) dataset. This project aims to experiment with and review the fundamentals of machine learning and neural network training I've learned.

Not only does this serve as a practice project for machine learning, I also tried my best to follow some good coding practices I've learned these years, such as regression testing, SOLID, writing documentation, etc. I would say this might be the conclusion of my past two years in CS. Looking back, I think I've improved a lot. At least I'm not creating gibberish that would break with a single touch (gibberish pro max?).

If you find any mistakes or anything that I should improve, please let me know. It would help me a lot.


## Datasets
### Iris Dataset
The Iris dataset is a well-known dataset in the machine learning community, containing 150 samples of iris flowers from three different species. Each sample has four features: sepal length, sepal width, petal length, and petal width.

The data contains two csv files, the input (`./data/iris/iris_in.csv`) and the output (`./data/iris/iris_out.csv`).

### ORL Dataset
The ORL (Olivetti Faces) dataset contains 400 grayscale images of 40 individuals. Each individual has 10 different images, and each image is 32x32 pixels.

The data had been split into two folders, the training dataset and the testing one (50% each).
You can put them back into one folder and use the `DataSplitter` class if you are planning to use another train/ test ratio. (P.S. I somehow got the dataset with them separated.)

## Usage
### Setup
The required packages are in `./requirements.txt`.

### Running Experiments
> I'm a bit lazy so no argparse. Sorry for the inconvenience.
> I'll leave comments so it will be easier to modify.

There are five experiments in total. 3 of them use iris, and the other 2 use ORL. They will be explained in [Experiments](#experiments).

Iris Dataset Classification:
```
python ./lab1.py
python ./lab2.py
python ./lab3.py
```
ORL Dataset Classification:
```
python ./lab4.py
python ./lab5.py
```
Load saved models:
Please see `./load_model.py`.

Optimize Path Comparison:
```
python ./compare_quad.py
python ./compare_dent.py
python ./compare_lr.py
```

### Configuration
> All the defualt configurations are in the code.


## Experiments
You can run the code and see the results for yourself. It is quite consistent (except exp1 which is bs).
Below is the description of the experiments and my thoughts on the results.

> Iris
### Experiment 1: A naive approach to classification
In this experiment, we use regression to solve this classification task. What we do here is to set the output node count to 1 and set thresholds for classification. We set the threshold to 1.5 and 2.5, hoping that the model can "miraculously" sort things out. 

> P.S Three intervals, three classes. This is stupid. :(

After playing around with the model, I found that keeping the model linear is better than making it non-linear. Also, using only one layer yields the best result. (If linear regression suffices, do we need to create a neural network?) This is a good baseline model for us to start with.

### Experiment 2: Improved regression
This time, we increase the output node count to 3, each representing a class. We encode the target using one-hot, and we use softmax as the last activation function.
The goal here is to optimize the output to be as close as the encoded target matrix. We take the mean square loss as the criterion and optimize the model accordingly.

The results are pretty good! But this will become an issue when the task becomes harder (I guess).
I wasn't expecting it to do this good (hitting 85+% consistently). The model is doing regression, not classification, after all.
I might try this on another dataset in the future. Quite interesting.

### Experiment 3: Good o' cross entropy
Cross entropy. I don't think I need to explain it any further.

The results aren't even surprising. There are some things worth mentioning though.
The model used here is almost the same as in [Experiment 2: Improved regression](#experiment-2-improved-regression). The only difference is the last layer: this one doesn't use softmax (as softmax is "integrated" in the cross entropy loss class). It outputs the results from the last linear layer. By setting the same hypermeters for both experiments, we can compare how these two methods work.

The difference isn't obvious, but we can still see that cross entropy works better.
As I said in the previous experiment note, I think I might do this experiment on a larger dataset.

> ORL
### Experiment 4: PCA + network
The ORL dataset consists of 32x32 (don't ask me why, I know it should be something like 92x112, but the main goal of this resp is to implement machine learning components, not this).

Here, we first flatten the image into a 1x1024 array. Then we apply PCA for dimension reduction, leaving only the essential components.
After that, we normalize the data using max-min norm.
Lastly, everything is fed to a cross entropy classifier, which sorts out everything.

Welp. Without a baseline, I can't really say anything about it.
But this is still something worth doing. It gives me a concrete view of what eigen-everything is. It's not just some abstract thoughts my linear algebra prof scribbles all over the board. I wish someone would show me this earlier. It would have helped a lot.

### Experiment 5: PCA + LDA
Instead of taking the max-min norm and feeding it to a nn classifier, we apply LDA.
The model predicts the output by taking euclidean distance between the class mean and the input "in the LDA space" (not sure if that is the correct term for this), and then finding the closest.

I'm glad I didn't forget everything from linear algebra.


> Optimize path comparison
### Experiment 6: Comparison on a quadratic surface
This is actually just an experiment to test if plotting something like this is even possible with my framework.
I didn't have this in mind when I was writing most of the components. Fortunately, it worked!

It's straightforward when you run the script. Feel free to play around with the parameters (`main` in `./compare_quad.py`).

### Experiment 7: Comparison on a surface with dents
This one is interesting.
I made the surface so that there are two dents on it, but the dents are actually local minimums.
I know it's cheating (because with a different set of parameters, the results vary), but it can clearly show the differences between optimizing algorithms.

### Experiment 8: Different learning rates
This experiment shows how different learning rates effect the converging process.


## Results
The results of experiment 1-5, including the saved model and the results, are saved in `./saved_model/`.
For experiment 6-8, please run the respective scripts.