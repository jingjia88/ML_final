# Machine Learning Final Report
Team: オオカミ君
Members: R10944050 洪靖嘉, B08901033 王瀞桓, B08901132 任瑨洋

## I. Data Preprocessing
We have four data sets mainly: ```demographics.csv```, ```location.csv```, ```satisfaction.csv```, and ```service.csv```. We consider each data set as a "questionnaire". After some observation, we discovered that not all customers in the train_ID / test_ID have filled out all four questionnaires, so it is crucial to enable our model to tolerance big data loss. Also, there are some unknown customers in those data sets that are not in train_ID, nor in test_ID. Of course they have no ```status.csv``` results, so we can't put these data directly into our training machine. These "useless" data should be clean in proper time in order to not influence our data set during training.
After some observation, we discovered that each column has about 770 data missing, about 12.5% of the whole column. This seems huge, but we also discover that there are some redundant data that can help us recover some of the data.
Also note that there are some non-scalar features in the questionnaire, such as ```City``` in ```location.csv``` and ```Internet Type``` in ```service.csv```. Thus we choose some methods, such as one-hot encoding, to encoding them. Below we will introduce our method in each questionnaire.
### A. demographics
1. For ```Gender``` and other Yes/No problems, such as ```Under 30```, ```Senior Citizen```, ```Married```, and ```Dependents```, we can encode these feature by simply using $-1, 0, 1$.


| Encoding Results | Male | Female | Unknown |
| -------- | -------- | -------- | - |
| Gender     | 1     | -1     | 0 |

| Encoding Results | Yes | No | Unknown |
| -------- | -------- | -------- | - |
| Yes/No Problems     | 1     | -1     | 0 |

2. ```Age```, ```Under 30``` and ```Senior Citizen``` are related. By using ```Age``` we can decide the other two feature directly without any possible loss. For missing ```Age```, we can recover that part with a higher precision then usual by refering the other two known features. Here are our detailed strategies:

| Under 30 | Senior Citizen | Proposed Value |
| -------- | -------- | -------- |
| Yes     | No     |$\text{avg}[\text{customer}_{\text{Under 30}\ \&\ !\text{Senior Citizen}}.\text{Age}]$|
| No     | No     |$\text{avg}[\text{customer}_{!\text{Under 30}\ \&\ !\text{Senior Citizen}}.\text{Age}]$ |
| No     | Yes     |$\text{avg}[\text{customer}_{!\text{Under 30}\ \&\ \text{Senior Citizen}}.\text{Age}]$ |
| Yes     | Unknown     |$\text{avg}[\text{customer}_{\text{Under 30}}.\text{Age}]$|
| No     | Unknown     |$\text{avg}[\text{customer}_{!\text{Under 30}}.\text{Age}]$|
| Unknown     | Yes     |$\text{avg}[\text{customer}_{\text{Senior Citizen}}.\text{Age}]$|
| Unknown     | No     |$\text{avg}[\text{customer}_{!\text{Senior Citizen}}.\text{Age}]$|
| Unknown     | Unknown     |$\text{avg}[\text{customer}.\text{Age}]$|

3. ```Dependents``` and ```Number of Dependents``` are related, since we can fully recover missing ```Dependents``` by known ```Number of Dependents```. As for missing ```Number of Dependents```, we can also recover part of it by refering ```Dependents```:

| Dependents | Proposed Value 
| -------- | -------- | 
| No     | $0$     |
| Yes  |$\text{avg}[\text{customer}_{\text{Dependents}}.\text{Age}]$|

### B. location
1. ```Count```, ```Country``` and ```State``` are not important, since they only have one meaningful unique value (which is 1, USA, and Califonia, respectly).
2. Since the case is about churns in a telecom company we can guess that the churn is related to regions, and population of those certain regions. Thus, we need to transform ```City```, ```Zip Code```, ```Lat Long```, ```Latitude```, and ```Longitude``` into meaningful regions and population. Since we have ```population.csv``` which shows the population of each zip code region, our first task is to recover ```Zip Code```.
    1. Since latitude and longitude gives a lot of geographic information, we first recover ```Lat Long```, ```Latitude``` and ```Longitude``` by just simply split the strings.
    2. Since if two coordinates are close, there is a high chance that they live in a same region, that is, they share the same zip code. Thus, for most of the customers with unknown zip code, we assign their zip codes as their nearest neighbours' zip codes. If their living city is known in the data set, we will add this criteria as a filter, just for sure.
    3. For the rest of the customers, we use an python library called ```geopy```. The module can convert latitiude and longitude to addresses, including zip codes. Since ```geopy``` is our last chance, we delete certain data if ```geopy``` cannot distinguish the zip code correctly.
3. After ```Zip Code``` is recovered, we discover that there are too much different zip codes, and there are too less customers in each zip code regions. Thus, we decided to use ```City``` as our region feature, and use zip code to recover ```City```. It's easy since customers with the same zip code will live in the same city.
4. Next we tried to encode ```City```. One-hot encoding is impossible since there are 1000+ different cities. For models which can tolerance highly non-linear features, encode those cities by simply ordinal encoding is possible. But since we don't like such high non-linearality, we tried to do **target encoding** to encode the cities.
    1. Target encoding is the process of replacing a categorical value with the mean of the target variable. (cite: h2o.ai)
    2. But in our case, our target is not a scalar. Instead, it's a category, which means that it doesn't have a mean.
    3. To "produce" the mean, we do **one-hot encoding** on the target. This splits the output from one to six.
    4. For each "output" we do target encoding on the cities. Thus, for each "output", each city gets a scalar value. And we can use six values to represent a city.
5. At last we convert the ```Zip Code``` into population by using the data in ```population.csv```. And then our processed ```location.csv``` is done: 6 dimensions to describe a city, and one dimension to describe the population in that certain area.
### C. satisfaction
1. Since there are only one dimension in ```satisfaction.csv``` , we had no chance to recover it. Thus, for lost data, the only way to process them is to delete them.
2. Also note that the value in ```satisfacion.csv``` is $1-5$, while the required result is category $0-5$. It's almost impossible to train only using the data in this questionnaire, thus we often combine this dimension with other questionnaires.
3. Note that when combining this questionnaire with others, we encounter some customers that filled ```satisfaction.csv``` but not the other questionnaire, or filled the other questionnaire but not ```satisfaction.csv```. Thus we developed four strategies to recover those loss:
    1. Delete the data if one of the questionnaire is empty.
    2. Fill in $\text{mean}(satisfaction)$ if customers in other questionnaire did not fill out ```satisfaction.csv```.
    3. Fill in $\text{mode}(satisfaction)$ instead.
    4. Fill in $0$ instead.
### D. service
1. For Yes/No problems, we encode them as $-1, 0, 1$ as usual.
2. For ```Number of Referrals``` and ```Referred a Friend```, their relationship is similar with the case ```Dependents``` and ```Number of Dependents``` in ```demographics.csv```. ```Phone Service``` and its related terms, and ```Internet Service``` and its related terms also have the similar relationship. Thus we apply the same process to those dimensions to recover the data.
3. We noticed that some of the dimensions are related. For example, ```Tenure in Month``` times ```Monthly Charge``` approximately equals to ```Total Charges```, and ```Tenure in Month``` times ```Avg Monthly Long Distance Charges``` equals ```Total Long Distance Charges```. Also, ```Total Charges + Total Extra Data Charges + Total Long Distance Charges - Total Refunds = Total Revenue```. We often recover those data by these formulas, and delete the data if there are too much loss.
4. For categorical dimensions such as ```Offer```, ```Internet Type```, ```Contract```, and ```Payment Method```, we apply one-hot encoding since there are only a few unique values, and the result of one-hot encoding is affordable.

## II. Our Three Models

### A. SVM
#### a. Idea
From some basic logic we can gain some knowledge, such as churn category ```competitors``` might related to regions, ```dissatisfaction``` and ```attitude``` might related to ```service.csv``` and ```satisfaction.csv```. Thus, my plan is to train models for ```demographics.csv + satisfaction.csv```, ```location.csv + satisfaction.csv```, and ```service.csv + satisfaction.csv```, one model each by using SVM. Then combine the three models to decide the final prediction.
The reason I chosed SVM is beacause I can expected that those churn categories should be group in piles, such as if in one region there are more competitors, the churn reason will more likely to be ```Competitors```. Since SVM is lightweighted, very efficiency, easy to implement, and good at grouping, I would like to give it a try.
#### b. How to build each model
1. **Bagging**
    * **Probability distribution**: Each model will be fed with one of the data sets. We apply bagging to generate our training data sets for training the model, and generate validating data sets to validate our submodels, in the purpose of generating the combined model. However, we found out that distribution of the data set is not even, that is, the portion of ```No Churn``` data point is much more than the other categories. We tried apply the data set directly to our training, and the model decided to predict everything as ```No Churn``` since the "accuracy" is high for this certain strategy. Therefore, we have to adjust the probability distribution of our bagging in order to generate a more balanced data set. When doing bagging we need to choose the probability of the data points inversely proportional to the frequency of category. For example, if the distribution of each category is like the chart below:
        | No Churn | Competitor | Dissatisfaction | Attitude | Price | Other |
        |----------|------------|-----------------|----------|-------|-------|
        |   0.4    |     0.1    |       0.1       |    0.2   |   0.1 |   0.1 |

        We notice that the ```No Churn``` category takes much more portion in the dataset, then we will set a lower probability to choose ```No Churn``` data point. By doing so, we will choose each category data point evenly after bagging, that is, generate a more balanced training set.
    * **Training data set**: In my SVM senario, the size of training data set is in the size of $0.4 \times \text{size}(\text{Total Training Data Set})$, duplicated samples are allowed. The number $0.4$ is relatively small to usual training, this is because that I discovered that the size of $\text{Category} = {4, 5, 6}$ is very small. If I want a large data set with every category uniformly distributed, those data will be repeated over and over again. This will cause certian bias when evaluating $E_{in}$, thus the performance of the submodels will not be good enough. $0.4$ is a descent number that the repetitions of the $\text{Category} = {4, 5, 6}$ will no be too overwhelmed, and the total size of data set is not too small.
    * **Validating data set**: In my SVM senario, I cannot choose the other not-chosen data easily as my validation set. The reason is that almost most of the $\text{Category} = {4, 5, 6}$ are chosen in the training data set. If I chose those not-chosen data as my validation data set, my validation data set will be extremely unbalanced, and this is not what I hope. Instead, I choose those validation data set by bagging from the total training data set, with the same probability distribution, but not allowing repeated samples. The probability distribution is to guarantee balanced, and not allowing repeated samples is to force the sampler to choose a variety kinds of data, trying to make up the cons caused by the above bagging.
2. **SVM with different $(C, \gamma)$**
    * I use the python package ```libsvm```, and choose C-SVC with radial basis function as my kernel. This is something familiar in our class, and radial basis function guarantees high-degree transformation with descent weights.
    * I used $C = [0.5, 1, 2, 5, 10, 20]$ and $\gamma = [0.1, 1, 10, 100, 1000, 10000]$. Among thse $C$s and $\gamma$s I find the best $(C, \gamma)$ pair that produces the best accuracy in the validation set.
3. **Re-train**
    * To enjoy the benefits of whole data set, I will re-train my model by using the whole data set with the $(C, \gamma)$ chosen. Note that since the original data set is still unbalanced, the data set I used here is also produced by bagging, but with a bigger size ($0.6$).
#### c. How to combine the three models
1. By specifying "-b 1" in ```libsvm``` my model can output the probability estimate.
2. For one customer, if he/she filled out all the questionnaires, then my three submodels will output three probability estimates. They are $g_1, g_2, g_3 \in \mathbb{R}^6$, respectly.
3. My plan is to predict the results by $g_1 + g_2 + g_3$, element-wise, and choose the one category with the largest probability estimate.
    * Originally what I want to do is $\sum \alpha_i g_i$. The method above is apparently a special case that $\alpha_i = 1$. But we discovered that:
        1. $\alpha_i$ is too hard to obtain with "proper methods". Of course we can obtain it by something like gradient descent, but the performance may not be good. As for those "good models", it's too hard for us to implement the method.
        2. Actually $\alpha_i = 1$ is not a bad choice. Since if the model itself is better, then it will produce a higher probability estimate then other "bad" models, and don't need an additional constant to boost it. Thus, $\alpha_i$ is not that crucial overall.
    * If the customer didn't fill out one of the questionnaire, the certain $g_i$ remains zero. Thus, my model can tolerance this type of loss.
#### d. Results
1. First we have the results from different "satisfaction fill-in strategies". The below chart is with train data set size $0.4$, validation set size $0.2$, and overall training data set size $0.6$ (for training the model after $(C, \gamma)$ is chosen). We can see that the best strategy is **mode**.
    * Note that the "Accuracy" is calculate over the whole train data set (by each csv), without bagging.

|  |  | Delete | |
| - | - | - | - |
|           | Loca.  | Demo. | Serv. |
| $C$       | 0.5    | 1     | 2     |
| $\gamma$  | 10     | 1     | 0.1   |
| Accuracy* | 91.50% | 80.33% | 93.71% |
| Kaggle F1 (private) | | 0.25375 | |

|  |  | Mean | |
| - | - | - | - |
|           | Loca.  | Demo. | Serv. |
| $C$       | 0.5    | 0.5   | 0.5   |
| $\gamma$  | 100    | 100   | 0.1   |
| Accuracy* | 91.16% | 81.20% | 93.48% |
| Kaggle F1 (private) | | 0.23998 | |

|  |  | Mode | |
| - | - | - | - |
|           | Loca.  | Demo. | Serv. |
| $C$       | 0.5    | 5     | 0.5   |
| $\gamma$  | 10     | 1     | 0.1   |
| Accuracy* | 87.11% | 73.37% | 94.00% |
| Kaggle F1 (private) | | **0.27257** | |

|  |  | Zero | |
| - | - | - | - |
|           | Loca.  | Demo. | Serv. |
| $C$       | 20     | 10    | 0.5   |
| $\gamma$  | 1      | 1     | 0.1   |
| Accuracy* | 88.14% | 77.47% | 94.11% |
| Kaggle F1 (private) | | 0.26712 | |

2. In the below tables I adjust the size of the bagging train data set. We can see that $0.4$ is the best choice.

|  |  | $0.2$ | |
| - | - | - | - |
|           | Loca.  | Demo. | Serv. |
| $C$       | 0.5    | 0.5    | 0.5   |
| $\gamma$  | 10     | 10000  | 10    |
| Accuracy* | 87.78% | 81.79% | 87.11% |
| Kaggle F1 (private) | | 0.23512 | |

|  |  | $0.4$ | |
| - | - | - | - |
|           | Loca.  | Demo. | Serv. |
| $C$       | 0.5    | 5     | 0.5   |
| $\gamma$  | 10     | 1     | 0.1   |
| Accuracy* | 87.11% | 73.37% | 94.00% |
| Kaggle F1 (private) | | **0.27257** | |

|  |  | $0.6$ | |
| - | - | - | - |
|           | Loca.  | Demo. | Serv. |
| $C$       | 0.5    | 2   | 0.5   |
| $\gamma$  | 10     | 1   | 1   |
| Accuracy* | 87.90% | 72.96% | 94.57% |
| Kaggle F1 (private) | | 0.24542 | |

|  |  | $0.8$ | |
| - | - | - | - |
|           | Loca.  | Demo. | Serv. |
| $C$       | 1     | 10    | 0.5   |
| $\gamma$  | 10      | 1     | 1   |
| Accuracy* | 87.33% | 71.85% | 94.74% |
| Kaggle F1 (private) | | 0.25694 | |

3. History highest private score on Kaggle (since bagging requires randomness, and sometimes there are luck problems): **0.28849**

### B. Random Forest
#### a. Idea 
After data preprocessing we get three data sets which are ```demographics_and_satisfication.csv```, ```location_and_satisfication.csv``` and ```services_and_satisfication.csv```. I want to imitate the concept of ensemble learning. I train three models based on the above three data sets repectively and get three random forest models. Those threes models are viewed as weak classifiers and I want to combine the result of the three models to get better results.  
#### b. How to build each model
First I did some bagging, using the same method in (A.SVM-b-1) to ensure a more balanced data set. After bagging, the training data set is fed into he random forest model. Initially, I set the number of decision tree to 10 and use entropy for binay decision. Moreover, I also normalize the training data set and validating data set before training.  
#### c. How to combine the three models
We have two ways to combine the three models.
##### 1.
After training the model, we will feed the validating data set to get the probability distribution belongs to each class. For example, input (x , y) and get the output ([0.1 , 0.2 , 0.1 , 0.4 , 0.1 , 0.1] , y). We will collect the output for training the combined model. So now, we have 3 training data set from the three models respectively, and we want to combine the three data sets to make each data point has 18 features. The method as follows: assume y corresponds to x1 = [0.1 , 0.2 , 0.1 , 0.4 , 0.1 , 0.1] , x2 = [0.2 , 0.3 , 0.1 , 0.2 , 0.1 , 0.1] , x3 = [0.3 , 0.1 , 0.1 , 0.2 , 0.1 , 0.2], and then we will concatenate x1, x2 and x3 to x'=[0.1 , 0.2 , 0.1 , 0.4 , 0.1 , 0.1 , 0.2 , 0.3 , 0.1 , 0.2 , 0.1 , 0.1 , 0.3 , 0.1 , 0.1 , 0.2 , 0.1 , 0.2]. If y doesn't exist in some data set, then we will set x = [0 , 0 , 0 , 0 , 0 , 0] to concatenate with other x Finally, (x' , y) will be the data point to train the combined model.
##### 2.
The second way we will also combine x1, x2 and x3 to train the model. However, we let x' = x1 + x2 + x3. For example, x1 = [0.1 , 0.2 , 0.1 , 0.4 , 0.1 , 0.1] , x2 = [0.2 , 0.3 , 0.1 , 0.2 , 0.1 , 0.1] , x3 = [0.3 , 0.1 , 0.1 , 0.2 , 0.1 , 0.2], then x' = [0.6 , 0.6 , 0.3 , 0.8 , 0.3 , 0.4]. Similarily, if y doesn't exist in some data set, then we will set x = [0 , 0 , 0 , 0 , 0 , 0].
|  | 0   | 1   | 2   | 3   | 4   | 5  | 
|--|-----|-----|-----|-----|-----|----| 
|x1| 0.1 | 0.2 | 0.1 | 0.4 | 0.1 | 0.1|
|x2| 0.2 | 0.3 | 0.1 | 0.2 | 0.1 | 0.1|
|x3| 0.3 | 0.1 | 0.1 | 0.2 | 0.1 | 0.2|

original

|  | 0   | 1   | 2   | 3   | 4   | 5  |6|7|8|9|10|11|12|13|14|15|16|17|
|--|-----|-----|-----|-----|-----|----|-|-|-|-|--|--|--|--|--|--|--|--|
|x'|0.1 | 0.2 | 0.1 | 0.4 | 0.1 | 0.1 | 0.2 | 0.3 | 0.1 | 0.2 | 0.1 | 0.1 | 0.3 | 0.1
| 0.1 | 0.2 | 0.1 | 0.2|

method1

|  | 0   | 1   | 2   | 3   | 4   | 5  |    
|--|-----|-----|-----|-----|-----|----|
|x'|0.6 | 0.6 | 0.3 | 0.8 | 0.3 | 0.4|

method2

#### d. Result
In general, method 2 has better performance by 0.03(acc) than method 1, so I will only discuss method 2 here. After experiment, I found out that the number of decision tree in each weak classifier and in combined model will affect the accuracy.
|  | 20   | 30  |
|--|-----|-----|
|20|0.19228|0.19287|
|30|0.22553|0.24605|
|40|0.21328|0.21867|
|50|0.27320|0.23337|
|60|0.22716|0.20637|

column : number of decision tree in each weak classifier
row : number of decision tree in combined classifier
number in the sheet : accuracy

Generally, when number of decision tree in combined classifier reach 50, we will get better accuracy. Why acc in (30,30) is higher than acc in (50,30) I suppose because bagging don't give balanced data sets as expected. Since some category in the original data set only account for a little amount, so it's hard to sample those category.

### C. XGBoost
#### a. idea
XGboost is an algorithm that using gradient boosting framework, combining a set of weak classifier and delivers improved prediction accuracy. It could be used in both regression and classification problems. The reason of chosing XGBoost is because of the searching results. After referring to many kaggle cases, we found xgboost have a great performance in most cases. Considering the  advantages in Section 3, we decided to choose it as our last methods.


#### b. how to build model
We use package---xgboost to implement this model. After reading the preprocessed csv, we combine these data to one numpy array and impute ```np.nan``` for customer who doesn't have data in csv file. Then using ```LabelEncoder``` in sklearn to transform string features.  After this step, we could start spliting dataset to training dataset and testing dataset. 
Since the dataset in imbalance, status ```No Churn``` is much more than the other five status', we use ```SMOTE``` to solve the problem of data category imbalance. ```SMOTE``` is an improved way of synthesizing the method of randomly increase the number of samples in the minority class and randomly reduce the number of samples of the majority class. SMOTE would select datas that are close in the feature space, drawing a line between these data and creating a new sample at a point along that line. It doesn't sample in the data space, but in the feature space, so accuracy in SMOTE will be higher than the traditional sampling method. 
Traing data after oversampled would be put to XGBClassifier for training. We set parameters as: n_estimators=70, learning_rate=0.3,
num_class=6, objective='multi:softmax'. Then predicting data would be processed like training data, using ```np.nan``` for imputation and ```labelencoder``` for categorical features. 

#### c. result
by using ```classification_report``` in sklearn.metrics 
for test data = 0.05*all data
|  | precision|recall| f1-score|support| 
|--|-----|-----|-----|-----| 
|0| 0.98 | 0.97 | 0.98 | 156 | 
|1| 0.92 | 0.96 | 0.94 | 25 | 
|2| 0.78 | 0.78 | 0.78 | 9 |
|3| 0.91 | 1.00 | 0.95 | 10 |
|4| 0.80 | 0.67 | 0.73 | 6 |
|5| 0.83 | 0.83 | 0.83 | 6 |
|accuracy| | | 0.95| 212 |
|macro avg| 0.87 | 0.87| 0.87 | 212 |
|weighted avg| 0.95| 0.95 | 0.95| 212 |
*
On kaggle:
best private score: 0.31402 (corresponding public score: 0.28477)
best public score: 0.30341 (corresponding private score: 0.29266)

## III. Our Selection
For three approaches in Section 2, we would select XGBoost as our final recommendation.

### A. pros
1. It has many strategies to prevent overfitting, such as regularization terms, Column Subsampling, etc.
2. Fast execution speed since the support of parallelization at feature granularity. In our method, it only take no more than 20 sec.
3. Stop building trees  in advance to handle the situation that the prediction results are pretty good by using early stop. 
4. Having great performance on small and medium data.  
5. Allow missing values for features.
6. Have good scalability. Although XGBoost is sequential, it still scale very well when adding more and more threads.


### B. cons
1. Not suitable for processing high dimensional feature data.
2. Having worse performance on sparse and unstructured data compared to other methods.
3. Too many algorithm parameters. Tuning parameter becomes complicated.


## IV. Contributions
|洪靖嘉|王瀞桓|任瑨洋|
|-----|-----|-----|
|1. 幫我們這組拿最高分|1. Data Preprocessing|1. Data Preprocessing
|2. XGboost|2. Random Forest|2. SVM |


## References
* https://medium.com/data-design/investigating-xgboost-exact-scalability-d562b2b501c0
* https://www.datacamp.com/community/tutorials/xgboost-in-python