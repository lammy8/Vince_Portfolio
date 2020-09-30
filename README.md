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
### Titanic: Machine Learning from Diaster (Kaggle)
Enter github link here. Enter jupyter notebook link here.

Enter description here.

### Random Forest

#### Kyphosis

### XGBoost
<details>
<summary>Hourly Energy Consumption Forecasting</summary>

Enter findings and summary

</details>

### Support Vector machines

##### Breast Cancer

#### Iris Flower

## Linear Regression problems
### House Prices: Advanced Regression Techniques
Enter github link here. Enter jupyter notebook link here.

Enter description here.

### Loan Prediction
Enter github link here. Enter jupyter notebook link here.

Enter description here.

## Logistic Regression problems
### Advertising Data

### Titanic Logistic


## Natural lanugage procressing

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

### Bag of Words
Enter github link here. Enter jupyter notebook link here.

Enter description here.

### Spam text messages

### Yelp reviews

## Clustering
### Clustering with KMeans
Enter github link here. Enter jupyter notebook link here.

Enter description here.

University clustering

## Neural networks
Enter github link here. Enter jupyter notebook link here.

Enter description here.

Breast Cancer - TensorFlow

House Price Predictor - TensorFlow

Bank Note Authentication - TensorFlow

Loan Repayment Predictor - TensorFlow

Movie Review Text Classification - TensorFlow

<details>
 <summary><b>Rotten Tomatoes Movie Reviews Sentiment Analysis</b></summary>

Link to notebook: https://github.com/vincentlam13/code/blob/master/deep-learning/TensorFlow/movie-reviews-TF-text-classification.ipynb

Used TensorFlow neural networks to solve the Sentiment Analysis on Movie Reviews Kaggle competition. The dataset contains syntactic subphrases of Rotten Tomatoes movie reviews. The task is to label the phrases as positive or negative on a scale from 1 to 5. The aim is not label the entire review, but individual phrases from within the reviews, which is a more difficult task.

</details>

## Recommendations Systems
### MovieLens 

## Courses & Certificates
* Programming languages:
  * Intro to Python (April 2020) (365 Data Science)
  * [Automate the Boring Stuff with Python Programming](https://www.udemy.com/certificate/UC-4dd14984-5141-4d50-8d38-dfe7af4906b1/) (May 2020) (Udemy - Al Sweigart)
  
* Machine Learning:
  * [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/certificate/UC-70ca0a85-cd1a-487c-9795-7686a89c1827/) (June 2020) (Udemy - Jose Portilla)
