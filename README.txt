--Machine Learning Algorithm Model Module for Python 2.7--

The purpose of the algorithm model objects packaged in
model_classes.py is to provide a framework for rapid
prototyping and development of machine learning hypotheses
and applications in Python. The best use for this module
is as an aide to be imported to the interactive Python shell.

Each class instance, when initialized with a training set of
data and a few other parameters, will perform a different 
machine learning optimization algorithm to the data to produce
a hypothesis that produces a minimal amount of error. 

The class instance can then be called later with a cross
validation or test set of data to diagnose whether the 
hypothesis generalizes well. 

The design and implementation of these algorithms is based
on the lecture materials of Andrew Ng's Coursera course on
Machine Learning. 