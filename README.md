# Framingham_Heart_Study
Logistic Regression Analysis for Framingham Heart Study dataset

### Scope of study
- 1. Plots of various independent variables to understand the dataset and data patterns.
- 2. Data Cleaning to prepare data set for analysis
- 3. Implement Logistic Regression, Logit Model to explain the marginal effects of various variables on the probability of occurence of Coronary Heart Disease after 10 years.
- 4. Confusion matrix with lowest false positives and false negatives.

### Deadline - Dec 16, 2022 

## Explaining Dataset ##

The dataset is publically available on the Kaggle website, and it is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has a 10-year risk of future coronary heart disease (CHD). The dataset provides the patients’ information and includes 4,238 records,16 columns and 15 attributes. 

The goal of the dataset is to predict whether the patient has a 10-year risk of future (CHD) coronary heart disease.
Each attribute about the patient is a potential risk factor. There are both demographic, behavioral, and medical risk factors.

### Demographic:
- Sex: male or female (Nominal)
- Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)

### Behavioral
- Education: 0: Less than High School and High School degrees, 1: College Degree and Higher
- Current Smoker: whether or not the patient is a current smoker (Nominal)
- Cigs Per Day: the number of cigarettes that the person smoked on average in one day. (can be considered continuous as one can have any number of cigarettes, even half a cigarette.)

### Medical (history)
- BP Meds: whether or not the patient was on blood pressure medication (Nominal)
- Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
- Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
- Diabetes: whether or not the patient had diabetes (Nominal)

### Medical (current)
- Tot Chol: total cholesterol level (Continuous)
- Sys BP: systolic blood pressure (Continuous)
- Dia BP: diastolic blood pressure (Continuous)
- BMI: Body Mass Index (Continuous)
- Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
- Glucose: glucose level (Continuous)

### Predict variable (desired target)
- 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)
