# nb_dp
A Naive Bayes classifier with Local Differential Privacy incorporated with 5 LDP used for frequecy estimation.

Protocols|
---|
Direct Encoding|
Summation with Histogram Encoding|
Thresholding with Histogram Encoding|
Optimal Unary Encoding|
Symmetric Unary Encoding|

As part of this study, we prove that Local Differential Privacy techniques can be used to create good classification models using data which doesn’t compromise of the individual’s privacy. The individual’s data is perturbed before being sent to the data aggregator. The LDP encoding and perturbation techniques help to maintain the feature and class relationship even after perturbation thereby helping to create LDP classifier which performs as good as a normal classifier. We have re-created a LDP classifier using Naive Bayes classification based on the journal “Locally Differentially Private Naive Bayes Classification”, verified our model’s accuracy comparing our results with the original results in the paper and also used new data to compare it with other ML models. Overall, we achieved 90.21% accuracy for the Naive Bayes LDP classifier.
