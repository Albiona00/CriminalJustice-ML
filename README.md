# Justice Insight: Enhancing Criminal Justice Decision-Making through Predictive Analytics-ML

#### University and Course Information

University: University of Pristina
Faculty: Faculty of Electrical and Computer Engineering
Level: Master's
Lecture: Machine Learning
Lecturer: Prof.Dr.Ing. Lule Ahmedi, PhD.c Mërgim H. HOTI
Students: Albiona Vukaj and Rina Shabani

#### Dataset attributes:
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

#### Number of attributes: 14 
Size: 91598 rows
Dataset size: 11,485 KB
Source: Data.gov 
1 Dataset Link: https://catalog.data.gov/dataset/texas-department-of-criminal-justice-releases-fy-2022
2 Dataset Link: https://catalog.data.gov/dataset/texas-department-of-criminal-justice-receives-fy-2022

#### Overview
This project aims to predict:
Classification of Crime Types: The model can be used to classify crimes into different categories such as "Violence," "Drugs," "Other" based on the description and offense code.
Sentencing Time Analysis: Analyses can uncover trends in judgments based on historical sentencing data using date columns and sentencing years.
Parole Prediction: A model can determine the factors influencing decisions for parole and predict which cases are more likely to be approved for parole.
Detection of Discrimination Patterns: Data analysis can reveal potential discrimination patterns based on race or gender in sentencing or parole decisions.

#### Phase 1

#### Data types
Date: Numerical-Discrete	
Type: Categorical-Nominal	
Inmate Type: Categorical-Nominal	
Gender: Categorical-Nominal	
Race: Categorical-Nominal	
Age: Numerical-Discrete	
County: Categorical-Nominal
Offense Code: Categorical-Nominal	
Offense: Categorical - Nominal	
Offense Description:Categorical - Nominal	
Sentence Date: Numerical-Continuous	
Offense Date: Numerical-Continuous	
Sentence (Years): Categorical-Ordinal	
Record Type: Categorical-Nominal

- Check for Null Values:In handling null values based on the provided data, we first addressed the missing values in the 'Offense Description' column by filling them using the corresponding 'Offense Code'. Subsequently, any null values in the 'Sentence Date' and 'Offense Date' columns were replaced with 'Unknown'. This process ensured that the dataset was cleansed of null values, allowing for further analysis and modeling with complete data.

- Find and Remove Outliers: We detected outliers using Z-Score and and subsequently remove or adjust them to mitigate their potential impact on skewing the predictions of the model.

- Aggregation: We conducted an aggregation by calculating the duration between two key dates: the 'Offense Date' and the 'Sentence Date', resulting in a new column named 'Between Days'.Also we transformed the 'Sentence (Years)' column into numerical values and segmented it into ranges.

- Implement SMOTE (Synthetic Minority Over-sampling Technique)-to address imbalanced classes by oversampling the minority class, thereby balancing the dataset. This approach aims to improve the model's capacity to learn from minority instances effectively, with a specific focus on the 'Offense' class imbalance. We will evaluate the dataset's performance both before and after applying the SMOTE technique.

Before removing outliers and balancing classes-SMOTE algorithm

![1](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/eadd28d0-1193-4c63-bee2-d92822ef1bfc)

After removing outliers and balancing classes-SMOTE algorithm

![2](https://github.com/Albiona00/CriminalJustice-ML/assets/74986994/ad1996d5-8a50-4034-962e-449ea40bdc6d)

