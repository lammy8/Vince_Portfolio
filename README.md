# Data Science Portfolio | Vincent Lam
This portfolio is a collection of notebooks which I have created to further my knowledge on data analysis and machine learning algorithms. The projects have been categorised by type.

## Personal projects

<details>
 <summary><b>Climbing crag selector</b></summary>
 Link to project: https://github.com/vincentlam13/climbing-crag-selection
 
 - Created a tool to help decide which climbing crags to focus on for future climbing trips, with the goal to improve the climbing grades of my friends and I.
- Scraped over 4000 routes/problems for the climbing destination in question.
- I will update this repositry as I analyse more climbing destinations and update whether my efforts were successful.

### EDA
Below are a few insights gleamed from the routes analysis.

Breakdown of climbing disciplines in Dorset

![Climbing disciplines in Dorset](/images/climbing-disciplines.png)

Breakdown of sport grade distribution

![Sport grade distribution](/images/climbing-sport-grade-distribution.png)

7a routes sorted by number of logs

![7a routes sorted by popularity](/images/climbing-logged-7a.PNG)

I filtered out crags that had at least two 7a routes that had been logged 100 times on ukc.

![Chosen crags](/images/climbing-crags.PNG)


Based on this data analysis we will be going to Blacknor South, Winspit, and Cheyne Weares Area. The list of potential routes are shown in table below.

|  Crag | Route  | Star | Height (m)  | Notes  |
|---|---|---|---|---|
| Blacknor South  | Sacred Angel  | ** | 15  | Easy up to ledge, then fingery crux with pockets |
| Blacknor South  | To Wish the Impossible   | *** | 20  | Sustained with delicate & fingery climbing, lots of rests, big moves off jugs  |
| Winspit  | Peppercorn Rate  | **  | 20  | Tough and pumpy with a blind crack  | 
| Winspit  | Exuberence  |  * | 20  | One hard bit at top, not so many onsightsbut alot of redpoints  | 
| Winspit  | Ancient order of Freemarblers  | **  | 20  | Steep stamina climbing, decent proportion of onsights  | 
| Winspit  | Gallows' Gore  |  ** | 20  | Powerful start about a V3/V4, but high rate of onsight and redpoints  | 
| Winspit  | Agonies of a Dying Mind  |  * | 20  | Powerful start about a V3/V4, but high rate of onsight and redpoints  | 
| Cheyne Weares Area  | The Accelerator  | *  | 7  |  Sounds super soft and pump shouldn't be a factor! | 

### Future Improvements
- Automate the analysis process for future climbing trips, likely destinations include the Peak Districtm Southern Sandstone, Costa Blanca, and Chamonix.
- Screenshot and scrape the bar chart information on style of ascents and voting of the route diffulty, and sort the routes by highest percent that have been onsighted or by 'softness'. Example shown below.

 
 Please see below findings:
</details>

<details>
 <summary><b>Golf tournament predictor</b></summary>
 Link to project: https://github.com/vincentlam13/golf-tournament-predictor
 
 
 - Created a tool to predict likely winners of PGA tournaments.
- Scraped PGA stats website for useful determinative data.
- Created new metrics to predict winners based on domain knowledge.
- Created a script to send the DataFrame csv file to google sheets, using gspread (a Python API for Google Sheets).
- Created a function to test model against historic data.
 
 ## Testing Prediction Against Historic Tournaments

| Year  | Tournament  | Golfer  | Predictive Ranking  | Real Position  |
|---|---|---|---|---|
|  2019 | The Open Championship  | Brooks Koepka  |  11 |  4 |
| 2018  | The Open Championship  | Justin Rose  | 4  | 2  |
| 2017  | The Open Championship |  Jordan Spieth | 4  |  1 |
| 2017  | US Open  |  Rickie Fowler | 2  |  5 |
|  2016 | The Open Championship  | Sergio Garcia  | 6  | 5  |
|  2014 | PGA Championship  | Rory Mcllroy  | 1  |  1 |

</details>


<details>
 <summary><b>COVID-19 analysis</b></summary>
Interactive visualisations using plotly library.

Used data from European Centre for Disease Prevention and Control to analyse trends around the world and in the UK.

Visualisation of the infamous 'R' value.
</details>

### London House Price Predictor


### London data scientist salary predictor
Enter text here.


## Classification problems

### Logistic Regression
<details>
 <summary><b>Titanic Survivor Logistic Classifcation</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/machine-learning/regression/logistic-regression/titanic-logistic.ipynb
 

This notebook was created in conjunction with the Data Science Bootcamp course. The aim of the notebook is to use logistic regression to classify whether or not a passenger on the Titanic survived based on passenger information.
</details>


<details>
 <summary><b>Advert Click Logistic Classifcation</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/machine-learning/regression/logistic-regression/advertising-data.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. The aim of the notebook is to classify whether or not a particular internet user clicked on an Advertisement. A logistic regression model will predict whether or not they will click on an ad based off the features of that user.
</details>


### Random Forest

<details>
 <summary><b>Kyphosis Random Forest Classification</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/machine-learning/classification/random-forest/kyphosis-with-decision-trees-and-random-forest.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. The aim of the notebook is to classify whether or not a child has Kyphosis, a spinal condition, based on their age in months and number of vertebrae involved in the operation. This notebook compares the results between a decision tree and random forest classifier.
</details>

### XGBoost
<details>
<summary>Hourly Energy Consumption Forecasting</summary>

Enter findings and summary

</details>

### Support Vector Machines

<details>
 <summary><b>Breast Cancer SVM Classification</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/machine-learning/classification/support-vector-machines/breast-cancer-SVM.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course.
Used Support Vector Machine classifier to predict whether a patient's breast cancer is benign or malignant based on the size of the breast tumour features. A gridsearch was incorporated to find the best parameters.
</details>



<details>
 <summary><b>Iris Flower SVM Classification</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/machine-learning/classification/support-vector-machines/iris-flower-SVM.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. This notebook uses these four features to predict what type of iris flower it is, using a support vector machine classifier.
</details>

## Linear Regression
<details>
 <summary><b>House Price Prediction</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/machine-learning/regression/linear-regression/US-housing-linear-regression.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. The aim of the notebook is to predict US house prices based on a number of features:
- Average area income
- Average area house age
- Average area number of rooms
- Average area number of bedrooms
- Area population
- Price
- Address

</details>


<details>
 <summary><b>Ecommerce User Experience Linear Regression</b></summary>
The link to this notebook: https://github.com/vincentlam13/code/blob/master/machine-learning/regression/linear-regression/ecommerce-linear-regression.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. An Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want. The company is trying to decide whether to focus their efforts on their mobile app experience or their website. This notebook aims to solve their problem.

</details>

## Natural Lanugage Processing

<details>
 <summary><b>Amazon Fine Foods Sentiment Analysis</b></summary>
 
 Link to project: https://github.com/vincentlam13/code/blob/master/natural-language-processing/sentiment-analysis/amazon-reviews-sentiment-analysis/amazon-reviews-sentiment-analysis.ipynb
 
 The purpose of this notebook is to make a prediction model that predicts whether a recommendation is positive or negative. This will be achieved by building a Term-document incidence matrix using term frequency and inverse document frequency.
 
 The performance of three machine learning algorithms were compared and visualised with a ROC curve:
 - Multinomial Naive Bayes Classifier
- Bernouli Naive Bayes Classifier
- Logistic Regression

 ![ROC Curve Classifier Comparison](images/amazon-sentiment-classifier-comparison.png)
 
 The ROC curve shows that the Logistic Regression Classifier provided the best results. Although the AUC value can be improved further. We shall focus on using logistic regression for the remainder of this notebook.
 
 #### Visualisation of sentiment analysis of food reviews
 
 ![Wordcloud of positive reviews](images/amazon-sentiment-wordcloud-useful.png)
 ![Wordcloud of negative reviews](images/amazon-sentiment-wordcloud-useless.png)
</details>


<details>
 <summary><b>Yelp reviews classification</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/natural-language-processing/yelp-reviews-NLP.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. The aim of the notebook is to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews. 
</details>

<details>
 <summary><b>Spam text messages classification</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/natural-language-processing/spam-sms-NLP.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. The aim of the notebook is to classify SMS messages into whether they are spam or legitimate messages.
</details>

## Clustering
### Clustering with KMeans

<details>
 <summary><b>University clustering</b></summary>
The link to this notebook: https://github.com/vincentlam13/code/blob/master/natural-language-processing/spam-sms-NLP.ipynb

This notebook was created in conjunction with the Data Science Bootcamp course. The aim of the notebook is to cluster universities into being a private or public school. 
</details>

## Neural networks

### Regression

<details>
 <summary><b>House Price Predictor with TensorFlow</b></summary>

Link to notebook: https://github.com/vincentlam13/code/blob/master/deep-learning/TensorFlow/house-price-predictor-TF-regression.ipynb

This notebook was created in conjunction with the Data Science Bootcamp course. This notebook predicts US house prices using TensorFlow linear regression by using many housing features.

#### Geographical visualisation of house prices

The below figure shows that Seattle houses are more expensive when they are waterfront properties.

![Visualisation of house prices by coordinates](/images/house-tensorflow-geo-price.png)

#### House price predictions

The below figure shows how the top 1% houses are skewing the predictions. The mode could be retrained on only the bottom 99% of houses.

![House price prediction](/images/house-tensorflow-predictions.png)

#### Model losses

The figure below shows that the loss and validation loss plots are similar and have no spikes, this means that there can be further training without risk of overfitting to the training data.

![House price prediction](/images/house-tensorflow-losses.png)

</details>

### Classification

<details>
 <summary><b>IMDB Reviews Sentiment Analysis</b></summary>

Link to notebook: https://github.com/vincentlam13/code/blob/master/deep-learning/TensorFlow/movie-reviews-TF-text-classification.ipynb

Used TensorFlow neural networks to solve the Sentiment Analysis on Movie Reviews Kaggle competition. The dataset contains syntactic subphrases of Rotten Tomatoes movie reviews. The task is to label the phrases as positive or negative on a scale from 1 to 5. The aim is not label the entire review, but individual phrases from within the reviews, which is a more difficult task.

</details>

<details>
 <summary><b>Breast Cancer Classification using TensorFlow</b></summary>

Link to notebook: https://github.com/vincentlam13/code/blob/master/deep-learning/TensorFlow/breast-cancer-TF-classification.ipynb

This notebook was created in conjunction with the Data Science Bootcamp course.
Used TensorFlow neural networks to classify patients' breast cancer as benign or malignant based on the size of the breast tumours features. 
The TensorFlow model consisted of:
- Three layers, going from 30 nodes to 15 to 1
- The first two layers had a Rectified Linear Unit activation function, and the last was a sigmoid activation function
- The loss function selected was binary crossentrophy and the optimiser was Adam
- Earlystopping via validation loss was used to prevent further losses
- Overfitting was prevented by using dropout layers, to turn off a percentage of neurons randomly

#### Model Evauluation

![Evulation of model ](images/breast-tensor-results.PNG)
</details>


<details>
 <summary><b>Bank Note Authentication TensorFlow Classification</b></summary>
 
The link to this notebook: https://github.com/vincentlam13/code/blob/master/deep-learning/TensorFlow/bank-note-authentication-TF.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. The aim of this notebook is to predict whether or not a bank note is authentic or not based on the features of the bank note. The Bank Authentication dataset is from the UCI repository.

The data consists of 5 columns:
- variance of Wavelet Transformed image (continuous)
- skewness of Wavelet Transformed image (continuous)
- curtosis of Wavelet Transformed image (continuous)
- entropy of image (continuous)
- class (integer)

Where class indicates whether or not a Bank Note was authentic.
</details>


<details>
 <summary><b>Loan Lending Predictor with TensorFlow</b></summary>

Link to notebook: https://github.com/vincentlam13/code/blob/master/deep-learning/TensorFlow/Loan-lending-predictor-tensorflow.ipynb


This notebook was created in conjunction with the Data Science Bootcamp course. The aim of this notebook is to predict whether or not a new potential customer will be able to pay back their loan.

</details>

## Recommendations Systems
### MovieLens 

## Courses & Certificates

* Machine Learning:
  * [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/certificate/UC-70ca0a85-cd1a-487c-9795-7686a89c1827/) (June 2020) (Udemy - Jose Portilla)

* SQL:
  * SQL for Data Analysis (Udacity)
  
* PowerBI:
  * [Microsoft Power BI - Up & Running with Power BI Desktop](https://www.udemy.com/certificate/UC-02014d8f-874f-4e5c-8ec5-c6e5d602ac0f/) (September 2020) (Udemy - Maven Analytics)

* Programming languages:
  * Intro to Python (April 2020) (365 Data Science)
  * [Automate the Boring Stuff with Python Programming](https://www.udemy.com/certificate/UC-4dd14984-5141-4d50-8d38-dfe7af4906b1/) (May 2020) (Udemy - Al Sweigart)
