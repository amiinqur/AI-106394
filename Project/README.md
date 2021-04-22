# AI 394 SPRING 2021: Project #
### PROJECT MEMBERS ###
StdID | Name
------------ | -------------
**63855** | **Amin M. Quraishi** <!--this is the group leader in bold-->
63910 | Muhammad Ashar
63927 | Abdullah Afaque
<!-- Replace name and student ids with acutally group member Names and Ids-->


## Description ## 
Well, the first two tasks were quite easy, as we just had to create a folder and get ready for the upcoming tasks with the help of the work, we did in Assignment 1. Task 3 was not really a piece of cake; we had to apply six different convolutions on two different filters having three different sizes. The three different sizes were mentioned beforehand: 5x5, 7x7 and 9x9. The necessary information about the filters was also given in the instructions.

Initially, we were having grievous issues in applying the different techniques on the datasets and the filters, but then we had a long discussion with our course instructor regarding the issue. Fortunately, he fixed the issue and answered all the queries we had, and we were finally able to move ahead with the task and the project. The first technique that we applied was multinomialNB from scikit learn on our filters and convolutions. This technique uses Laplace smoothing for classification and calculating the accuracy. We applied it on the three convolutions and all the different sizes. It took a lot of time to get the results as there were so many data points and labels, but ultimately, we got the results. We tried the multinomialNB on all the three different models, multiple times to ensure accuracy. We got the highest accuracy with the 5x5 model, so we made a submission on Kaggle with that model and the accuracy that we got on Kaggle was around 0.8 from the multinomialNB model. 

The next technique that we looked upon and applied was Linear Regression classification using scikit-learn. We imported linear regression library in our code before proceeding further, by this line of code: from sklearn.linear_model import LinearRegression. We applied the technique on all the three models and we got the highest accuracy again with the 5x5 model, so we made a second submission with that model. The accuracy this time was 0.83 on Kaggle from the LinearRegression classification technique.

 The last two techniques that we implemented were, Support Vector Machine (SVM) and K nearest neighbors (kNN). We studied and worked on these techniques in our lab classes as well, so it was relatively easier to work with them as compared to the first two techniques mentioned above. 
 
We imported the SVM library from scikit-learn and went ahead on with the code. We applied the three models on SVM and we got the highest accuracy with the 9x9 model, so we uploaded a third submission with that model on Kaggle. We got a better accuracy than the previous models of 0.87 from the SVM classification technique.

The last technique that we implemented was kNN. Following the trend of the previous three classification techniques, we imported the technique’s library from scikit-learn so that we could move ahead with the task. After successfully importing the technique, we applied kNN on all the three convolution models. The highest accuracy we got was with the 5x5 model, therefore we made the fourth submission on Kaggle with that model. And to our surprise, we got the highest accuracy of 0.98 with kNN as compared to all the three other models we used!

Given more time to apply more techniques to achieve the perfect accuracy of 1, we would apply Stochastic Gradient Descent classification and Decision tree technique. We would learn about these techniques from the various sources available on the internet and then implement them in order to acquire the highest possible accuracy.


## Highest Kaggle Score ##
![Highscore](https://user-images.githubusercontent.com/66859283/115763848-4fa10b00-a3be-11eb-809f-fb0acd301af4.png)

