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

## 2.4 Exercises
### Conceptual
**1. For each of parts (a) through (d), indicate whether we would generally
expect the performance of a flexible statistical learning method to be
better or worse than an inflexible method. Justify your answer.**

(a) The sample size n is extremely large, and the number of predictors p is small.
- Better. With more sample size overfitting will not be that much af a problem. As the number of predictors is small, there is likely a high bias involved (because it simplifies the real population function). A flexible model can lower the bias.

(b) The number of predictors p is extremely large, and the number of observations n is small.
- Worse. With a small number of observations the more flexible model is likely to overfit the training data.

(c) The relationship between the predictors and response is highly non-linear.
- Better. More flexible models can estimate more complex functions.

(d) The variance of the error terms, i.e. σ2 = Var(ϵ), is extremely high.
- Worse. A more flexible model in this case will learn the random patterns that comes with the high error variance.

---


**2. Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide n and p.**

(a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and the CEO salary. We are interested in understanding which factors affect CEO salary.
- Regression. We are interest in learning more about the relations within the variable of interest and it's predictors, meaning that we are most interested in inference. n = 500; p = 3.

(b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.
- Classification (success or failure, two classes); prediction; n = 20; p = 13.

(c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the % change in the US market, the % change in the British market, and the % change in the German market.
- Regression; prediction; n = 52; p = 3.

---

**3. We now revisit the bias-variance decomposition.**

(a) Provide a sketch of typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error curves, on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The x-axis should represent the amount of flexibility in the method, and the y-axis should
represent the values for each curve. There should be five curves. Make sure to label each one.

(b) Explain why each of the five curves has the shape displayed in part (a).

---

**4. You will now think of some real-life applications for statistical learning.**

(a) Describe three real-life applications in which classification might be useful. Describe the response, as well as the predictors. Is the goal of each application inference or prediction? Explain your answer.
- **Predicting default from future borrowers.** Response: default or not; predictors: wage, age, past borrows, etc. Inference or prediction: we may want to only know if they are going to pay, or we could want to know its characteristics to learn which product to offer.
- **Subscription renewal.** Response: renewal; predictors: features used. Inference: we could take action to implement more of the things the clients want more. This way, we could increase the renewal rate and increase profits.
- **Spam emails.** Response: spam or not; predictors: language usage, domain sent, usage of upper case letters. Predictions: we only want to classify correctly the emails so they are automatically blocked.

(b) Describe three real-life applications in which regression might be useful. Describe the response, as well as the predictors. Is the goal of each application inference or prediction? Explain your answer.
- **Stock prices.** Stock price; market information; prediction: only need to know if it's going up or down.
- **Wages.** Wages; human capital features; inference: need to know what should be the focus for improvement, for maximum wage gains.
- **Predicting Inflation.** Inflation; market information about the items prices; prediction: need to know the inflation value so it is possible to create a more developed portfolio.

(c) Describe three real-life applications in which cluster analysis might be useful.
- **Personalized advertising.** More specific and engaging ads for the public.
- **Insurance Pricing.** More precise pricing, more expensive for people that are more prone to use the insurance.
- **Similar public on social media.** When you see more content that you like, it's likely that your engagement in the social media will increase, consequently increasing the SNS revenue.
---

**5. What are the advantages and disadvantages of a very flexible (versus a less flexible) approach for regression or classification? Under what circumstances might a more flexible approach be preferred to a less flexible approach? When might a less flexible approach be preferred?**

- Advantages of a very flexible approach:
    - Less bias: doesn't need to simplify too much the real function $f$;
    - Can model complex functions;
    - Can incorporate a high amount of features.

- Advantages of a less flexible approach:
    - Less variance: less prone to overfitting;
    - Faster computation.

- Disadvantages of a very flexible approach:
    - High variance: can learn too much about the random patterns displayed on the training data;
    - Needs more observations;
    - More computational resources required.

- Disadvantages of a less flexible approach:
    - High bias: the error from simplifying the model is higher.
    - Can't model complex functions;
---

**6. Describe the differences between a parametric and a non-parametric statistical learning approach. What are the advantages of a parametric approach to regression or classification (as opposed to a nonparametric approach)? What  are its disadvantages?**

- A parametric approach is when estimating the function $\hat{f}(x) = y$, parameters are assumed. For example: in a linear regression approach, you simplify the estimation by assuming that $y$ is linear dependent on $x$. That means we could write 
    - $y = \beta_0 \cdot x_0 + \beta_1 \cdot x_1 + \dots$ 
- This means that our estimationg is easier (advantage). However, we may incur in high bias error as the real function $f$ may not be only made of linear components of $x$ (disadvantage). Consequently, non-parametric models will likely have higher variance and lower bias than parametric ones.
---

**7. The table below provides a training data set containing six observations, three predictors, and one qualitative response variable. Suppose we wish to use this data set to make a prediction for Y when X1 = X2 = X3 = 0 using K-nearest neighbors.**
(a) Compute the Euclidean distance between each observation and the test point, X1 = X2 = X3 = 0.
- $3$; $2$; $\sqrt{10}$; $\sqrt{5}$; $\sqrt{2}$; $\sqrt{3}$;

(b) What is our prediction with K = 1? Why?
- Green. Because the closest training observations is green (5th).

(c) What is our prediction with K = 3? Why?
- Red. Because the within the 3 closes training observations, there are two reds and one green.

(d) If the Bayes decision boundary in this problem is highly nonlinear, then would we expect the best value for K to be large or small? Why?
- Small. Because when K gets higher, the decision boundary of the algorithmn gets closer to a line. The same way, when K goes to lower values, the decision boundary gets more non-linear.


