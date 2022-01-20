# 1 - Introduction
- Statistical learning is a set of tools that helps us understand data.
    - Supervised: Predict or classify an outputs using inputs.
        - eg. Using years of study to predict wages.
    - Unsupervised: Learn more about the data, using without having outputs.
        - eg. Discovering clusters within the data.

- Statistical learning is important because with more and more data, it's uses are now beyond the academia.

# 2 - Statistical Learning

## 2.1 What is Statistical Learning
Suppose we have our variable of interest $y$ and a set of features $X$. Statistical learning are tools that helps us to estimate the populational (and sometimes purely theorical) function $y = f(x) + \varepsilon$.

As it's not possible to discover the **real** function $f$, we estimate the function with $\hat{y} = \hat{f}(x)$.

### Why estimate $f$?

#### Prediction
When we have our features set and want to predict the future values of the response variable. In this case a tradeoff will likely always occur: model interpretability and prediction power.

When our estimated $\hat{f}$ is complex and it's not feasible to understand, we have what it's called a *black box*. In this case the only objective is to maximize the prediction power.

- When dealing with prediction, even with a theoretical perfect estimate of $f = \hat{f}$, there will always be some error called _irreductible error_. This error comes from the population function $f$, and per definition, is not a function o the features set $X$.

#### Inference 
Sometimes we want to deeply understand the causality between events and our variable of interest. In this case, we can't simply throw a _black box_ model in our data and hope to magicaly understand the process. So we choose simpler models to be able to interpret the results from it.

e.g. If a company that sells a renewable subscription to a software wants to know what are the events that are more likely to *cause* a renewal, it can't just say that a variable is the _cause_ beecause it helps the most to predict the outcome. For example, there may be a variable called _bugs reported_ and it is highly correlated with the subscription renewal. Should the company implement more bugs willingly? Probably not, what is most likely happening is that the users that uses the most the software, and have a higher necessity for it, reports more bugs because: they enmcounter more bugs or they need the bug removed to use the software properly.

In this case, we need to be careful with the model results so we don't make mistakes.

### How do we estimate $f$?
- We need to find a function $f$ that $Y \approx \hat{f}(X)$ for every (X,Y).

#### Parametric
Two-step approach: 1. Assume a form for $f$; 2. Estimate it's parameters.

How does it work? When defining a form of $f$, our problem is not estimating $f$ anymore, it is estimating $f$'s parameters. This way, our problem is simplified.

e.g. Assume that $f$ is linear in $X$: this way, we can write that $y = \beta_0 + \beta_1*x_1 + ...$ and simply estimate our $\beta$s.


#### Non-Parametric
In this case we do not assume a form for $f$. The models try to estimate a function $f$ that approximates as much as possible the data, without being too wiggle or too rough. The benefits are clear: without the assumption the models can adapt to be any function possible. When making assumptions, it is not possible to be 100% sure that the assumption we make is the correct form of $f$, as $f$ is not observable.

The major drawback is the necessity of a higher quantity of data — way more than the necessary for a parametric approach — to have accurate estimates.

### The Trade-Off Between Prediction Accuracy and Model Interpretability
Basically, it says that models are in between a line of interpretability and flexibility. Models that are inflexible have higher interpretability, like the linear regression. 

This way it is normal to think that when dealing with *prediction* it is preferable to choose the most flexible and not interpretable model as possible and in the case of *inference* the most inflexible model. 

However, it's not always that a flexible model outperforms a inflexible one, most of the time because 

### Supervised Versus Unsupervised Learning
When not _observing_ the output variable in our training data, we cannot train models for prediction or inference. However, it is possible to get a better understading of the relationships in the data with **Unsupervised Learning**. One possible tool is called *cluster analysis*, with the most famous model being the K-Means.

#### Semi-supervised learning
When there are observations with the target variable and without it in the same dataset. The solution here is using semi-supervised learning models so that it can incorporate the information of the training target variable values as well as the information of the observations without the target. 

This situation can happen when it is expensive to collect the variable of interest while it is cheap to collect the predictors.
***
### Regression Versus Classification Problems
In simple terms, regression means that our variable of interest is quantitative, while in classification problems it is qualitative. However, the difference is not alway clear: in a logistic regression (for qualitative targets), it can also be interpreted as predicting the probability of a class, that is quantitative.

## 2.2 Assessing Model Accuracy
*There is no free lunch:* No model outperforms every other in every single dataset. We have to test a lot of models to see which one performs better for that specific problem.

### Measuring the Quality of Fit
Most of the times, we will want to measure the quality of the predictions in a **unseen** situation. We don't really care how our model that predics stock prices performns on our training data.

When we have more flexible models (more degrees of freedom), our training performance gets better. However, there is a possibility of the flexible model learning patterns that comes purely from randomness. In this case, the model will perform really well in the training data while really bad in the test data. This case when a model is worse (assessed in the test data)  than a less flexible one, we call it *overfitting*.

### The Bias-Variance Trade-Of
It is possible to prove that:
- $E(y_0 - \hat{f}(x_0))^2 = Var(\hat{f}(x_0)) + [Bias(\hat{f}(x_0))]^2 + Var(\varepsilon)$

That is, the **expected test** MSE can be decomposed in 3 parts: the variance and the squared  *bias* of $\hat{f}(x_0)$ and the variance of the error $\varepsilon$. As the first 2 terms are non-negative, we can safely say that our *expected test MSE* can never be below the variance of the populational error $\varepsilon$.

Our objective (when it applies) is to minimize the expected MSE. The way we can do that is by having low variance and low bias.
- Variance: In statistical learning, we call define variance as the amout of change that happens to $\hat{f}$ for a given amount of change in the training data. This way, if a model changes a lot when there is a lot of change in the training data, we call this model a high-variance model. More flexible models tends to have higher variance.
- Bias is the error that is introduced when we transform a complex model to a simpler one. The linear regression, for exaple, simplifies the problem by assuming that $y$ is a linear function of $x$. Simpler models tends to have higher bias, because it makes stronger assumptions.

- **Key takeaway: Generally, flexible models have high variance and low bias, while simple models have low variance and high bias.**

The way we can minimize MSE is using relative changes. When an increase in flexibility doesn't decrease the squared bias more than the variance, then it is not worth it.

### The Classification Setting
We have to change some things when dealing with a classification problem instead of a regression one.

When measuring performance, we can't use the *MSE*, for example. That is because we are dealing with qualitative data. One way to measure performance is with the *training Error Rate* metric:
- $ER = \frac{1}{n}\sum I(y_i \ne \hat{y_i}) $

It basically calculates the percentage of rightfully classified observations.

#### The Bayes Classifier
A good classifier is one that minimizes the **test** error rate. We can prove that by a simple classifier that uses conditional probability (e.g. classifies a observation to the most likely class *given* the predictors values) minimizes, on average, the *test* error rate.

- $Pr(Y = j|X=x_0)$

The Bayes error rate is the error rate that is calculated using the Bayes Classifier. As it is the best classifier when minimizing the error rate, the *Bayes error rate* is an irreductible error.

#### K-Nearest Neighbors
So why not use the Bayes Classifier in every classification problem? The thing is, we don't have the conditional probabilities (e.g. we don't know how $Y$ is truly affected by the $X$ values). Therefore, the Bayes classifier serves as an unattainable gold standard against which to compare other methods.

One method that tries to **estimate** the conditional probabilities is *K-nearest neighbors* (kNN) classifier. The way it works is by defining a positive and odd integer $k$, and for every test observation, it finds the closest $k$ **training** observations and classifies the test observation as the majority of the $k$ evaluated observations classes.

The algorithm estimates the conditional probabilities using the sample distribution of the $k$ nearest observations, and the uses it like the probabilities of the *Bayes Classifier*.

As in the regression setting, the training and test errors are not strongly correlated. If we choose $k = 1$, thenour training error goes to zero, clearly showing a low bias and very high variance. So the level of flexibility is extremely important in the classification as well.
