# Justice Insight: Enhancing Criminal Justice Decision-Making through Predictive Analytics-ML

#### University and Course Information

University: University of Pristina
Faculty: Faculty of Electrical and Computer Engineering
Level: Master's
Lecture: Machine Learning
Lecturer: Prof.Dr.Ing. Lule Ahmedi, PhD.c Mërgim H. HOTI
Students: Albiona Vukaj and Rina Shabani

#### Dataset attributes description
1.	Date: The date on which a criminal justice event.
2.	Type: The type of criminal justice event.
3.	Inmate Type: A classification code that specifies the inmate's custody level or security requirement.
4.	Gender: The gender of the individual.
5.	Race: The racial or ethnic identity of the individual.
6.	Age: The age of the individual at the time the event was recorded.
7.	County: The county associated with the individual's case, either as the location of the offense or the jurisdiction of the court.
8.	Offense Code: A unique numerical code that identifies the specific offense.
9.	Offense: A category that broadly classifies the nature of the offense.
10.	Offense Description: A detailed description of the offense committed by the individual.
11.	Sentence Date: The date on which the court issued the sentence for the individual's offense.
12.	Offense Date: The date on which the individual committed the offense.
13.	Sentence (Years): The length of the sentence imposed on the individual.
14.	Record Type: The type of record, indicating the status of the individual in the criminal justice process at the time.

- Number of attributes: 14 
- Size: 91598 rows
- Dataset size: 11,485 KB
- Source: Data.gov 
1 Dataset Link: https://catalog.data.gov/dataset/texas-department-of-criminal-justice-releases-fy-2022
2 Dataset Link: https://catalog.data.gov/dataset/texas-department-of-criminal-justice-receives-fy-2022

#### Overview
This project aims to predict:
Classification of Crime Types: The model can be used to classify crimes into different categories such as "Violence," "Drugs," "Other" based on the description and offense code.
Sentencing Time Analysis: Analyses can uncover trends in judgments based on historical sentencing data using date columns and sentencing years.
Parole Prediction: A model can determine the factors influencing decisions for parole and predict which cases are more likely to be approved for parole.
Detection of Discrimination Patterns: Data analysis can reveal potential discrimination patterns based on race or gender in sentencing or parole decisions. We will use decision trees, random forests, support vector machines, gradient boosting models, naive Bayes, logistic regression, linear regression for prediction.

## Phase 1: Model Preparation
#### Used Libraries:
![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/d6e388cb-7501-4e2b-91b6-777bd0d28028)

#### Integration of two datasets
![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/fa62cf7e-3ef1-4628-8b9b-b13861648dc9)


#### Column renaming to achieve similarity for integration of these two datasets
![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/af74aea1-7043-4b25-8f24-a23d158fb93b)


#### Data types
![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/a9d741b9-c3b0-45f4-ac28-6a372dc912ac)

##### Datatypes description
- Date: Numerical-Discrete	
- Type: Categorical-Nominal	
- Inmate Type: Categorical-Nominal	
- Gender: Categorical-Nominal	
- Race: Categorical-Nominal	
- Age: Numerical-Discrete	
- County: Categorical-Nominal
- Offense Code: Categorical-Nominal	
- Offense: Categorical - Nominal	
- Offense Description:Categorical - Nominal	
- Sentence Date: Numerical-Continuous	
- Offense Date: Numerical-Continuous	
- Sentence (Years): Categorical-Ordinal	
- Record Type: Categorical-Nominal

#### Unique values
![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/f2af4036-98a9-467d-bece-74c8a0644296)


#### Duplicate rows handling
- In our dataset we had 91598 rows, from which 1 was a duplicate row, which we handled.
  ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/4e5ce3d0-3268-4702-b038-d541eadf873b)

- This is what our dataset's row look like after removing duplicate rows:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/3fd53a03-9829-4096-8310-0eafb76cecc9)

- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/b38ab94c-ef2f-4303-a873-b454c48e8690)


#### Handling missing values
- Number of missing values before handling:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/2414f766-7db2-4813-8fa8-303af9467fb6)

- For handling we used 'Offense Description' for each 'Offense Code' to fill missing values in the 'Offense Description' column, and any remaining null values are filled with 'Unknown'.
- Number of rows after handling missing values:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/6de68eba-d5ad-4e71-87e6-5691eced9671)


#### Data transformation
- We converted the values of the "Sentence (Years)" column into integers:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/9d4013c6-5069-4ff9-9258-f0dba92722fa)

#### Vectorization
- Vectorization of Gender and Record Type columns with Label Encoding:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/05f9433a-5381-4b64-a686-dfe178408607)

- Vectorization with One Hot Encoding Type, Inmate Type and Race columns:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/d3c1d067-28c4-4da2-947d-ccceb04d42e3)


- Now this is what our dataset's datatype look, after all the changes we've done:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/e6f5d6c7-9cb2-4d3f-8916-d19f0c091082)


#### Outliers Removing
- We had a total number of 3277 outliers, this is the distribution of numerical columns before and after removing outliers:
- - ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/27bf0e5a-5358-4c78-a7b4-2748b519e78a)
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/3bb5c694-741b-4ea7-bdb5-2e8b04a9bd3a)


- Check for Null Values:In handling null values based on the provided data, we first addressed the missing values in the 'Offense Description' column by filling them using the corresponding 'Offense Code'. Subsequently, any null values in the 'Sentence Date' and 'Offense Date' columns were replaced with 'Unknown'. This process ensured that the dataset was cleansed of null values, allowing for further analysis and modeling with complete data.

- Find and Remove Outliers: We detected outliers using Z-Score and and subsequently remove or adjust them to mitigate their potential impact on skewing the predictions of the model.

- Aggregation: We conducted an aggregation by calculating the duration between two key dates: the 'Offense Date' and the 'Sentence Date', resulting in a new column named 'Between Days'.Also we transformed the 'Sentence (Years)' column into numerical values and segmented it into ranges.

- Implement SMOTE (Synthetic Minority Over-sampling Technique)-to address imbalanced classes by oversampling the minority class, thereby balancing the dataset. This approach aims to improve the model's capacity to learn from minority instances effectively, with a specific focus on the 'Offense' class imbalance. We will evaluate the dataset's performance both before and after applying the SMOTE technique.

Before removing outliers and balancing classes-SMOTE algorithm

![1](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/eadd28d0-1193-4c63-bee2-d92822ef1bfc)

After removing outliers and balancing classes-SMOTE algorithm

![2](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/ad1996d5-8a50-4034-962e-449ea40bdc6d)

![3](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/efbb4f1d-4b08-460b-9fd6-0944306cb0be)


