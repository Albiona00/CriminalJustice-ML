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
#### Imported Libraries:
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

#### Handling null values
- In our dataset were found various types of columns with null values:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/e82ff7b7-0105-4c2d-bb44-877fc4f9eaa5)

-This is after we handled null values:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/f08c71f9-fd8b-4456-9a4b-dc2b0cd66cd6)


#### Handling missing values
- Number of missing values before handling:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/2414f766-7db2-4813-8fa8-303af9467fb6)

- For handling missing values of 'Offense Description' column we used 'Offense Code' to fill missing values and any remaining null values are filled with 'Unknown'.
- Number of rows after handling missing values:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/6de68eba-d5ad-4e71-87e6-5691eced9671)


#### Aggregation
- We conducted an aggregation by calculating the duration between two key dates: the 'Offense Date' and the 'Sentence Date', resulting in a new column named 'Between Days'.
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/2c658467-f1b4-4b21-95f5-b0a1f2f1b1fa)


#### Data transformation
- We converted the values of the "Sentence (Years)" column into integers:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/9d4013c6-5069-4ff9-9258-f0dba92722fa)

- Now this is what our dataset's datatype look, after all the changes we've done:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/e6f5d6c7-9cb2-4d3f-8916-d19f0c091082)


#### Outliers Removing
- Sentence numeric distribution before and after outliers removing:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/2189b39d-6ec3-41a5-8e9e-a892ed85ba3e)


#### Feature engineering
- From the "Sentence (Years)" column, we created two new columns: "Sentence numeric" and "Sentence scaled", where we transformed the data from a range into integer values:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/a02c2bfd-0a0f-4a7f-87e4-2e23b919df36)


#### SMOTE algorithm
- Before balancing classes-SMOTE algorithm:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/69791c2e-d7b7-40ea-83ea-5480b7983486)

- After balancing classes-SMOTE algorithm:
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/ffa3a316-c90b-4f0a-a18a-e1868dba49c8)

## Phase 2: Model Training

In this phase, we trained the model using three different algorithms like: Random Forest, Decision Tree and Logistic regression, assessing precision, recall, F1-score, and accuracy across various data split ratios: 70% for training and 30% for testing, 80% for training and 20% for testing, and 90% for training and 10% for testing.

- Case 1:
In Case 1, we utilize a 70/30 split for training and testing, respectively.
- ![438083186_435503565845535_551507096414140460_n](https://github.com/Albiona00/CriminalJustice-ML/assets/150968383/7d5ed3f0-5d90-4392-8ea6-1040c7d509f6)

- Case 2:
In Case 2, we utilize a 80/20 split for training and testing, respectively.
- ![438060621_804436214899443_3072103938635799100_n](https://github.com/Albiona00/CriminalJustice-ML/assets/150968383/f0c26e72-249a-4f06-97d7-e1396b7808bb)

- Case 3:
In Case 3, we utilize a 90/10 split for training and testing, respectively.
- ![image](https://github.com/Albiona00/CriminalJustice-ML/assets/150968383/000624ba-d585-4ff4-9577-d0d04110d6fe)



