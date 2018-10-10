# Detecting Semantically similar questions
Project for CS671: Natural Language Processing (Spring 2018)

Websites like quora provide a platform for Question and Answering across various subjects and fields. However, the websites also contain loads of similar questions which have different sets of replies although they seek same answer. We aim to develop an application which can detect whether two questions on quora are semantically similar. More formally we want a model which when given two questions `q_1`and `q_2`, predicts whether the questions are similar in terms of meaning or not. The criteria for two questions being semantically similar is that they seek same answer

+ `preprocess.py` : This creates word vector representation for all questions in data-set
+ `getVector.py` : Contains functions that return vector representation of a question
+ `train.py` :  Trains a neural network which takes as input and two questions and predicts whether they seek same answer or not
+ `test.py` : Code to test the model
