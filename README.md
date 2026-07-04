# Automated MLOps Pipeline for Wellness Tourism Package Purchase Prediction

## Business Context

**"Visit with Us"** is a leading travel company aiming to transform its customer engagement strategy through data-driven decision-making. With the launch of a new **Wellness Tourism Package**, the organization faces challenges in accurately identifying customers who are most likely to purchase the package.

The existing manual customer selection process is inefficient, inconsistent, and prone to human error, resulting in missed sales opportunities and ineffective marketing campaigns. To overcome these challenges, the company seeks to implement a **scalable, automated, and reliable predictive system** that enables precise customer targeting and continuous model improvement.

By adopting an **MLOps-driven approach**, this project integrates data preprocessing, machine learning model development, deployment, and CI/CD automation using **GitHub Actions**, ensuring operational efficiency and adaptability to evolving customer behavior.

---

## Objective

The primary objective of this project is to **design and deploy an end-to-end MLOps pipeline** that predicts whether a customer is likely to purchase the **Wellness Tourism Package** before being contacted by the marketing team.

As an **MLOps Engineer**, the responsibilities include:

* Automating data cleaning, preprocessing, and feature transformation
* Building, training, and evaluating a predictive machine learning model
* Implementing CI/CD pipelines using **GitHub Actions**
* Ensuring model scalability, reproducibility, and continuous improvement

This solution enables the business to make **data-driven marketing decisions**, optimize customer outreach, and improve customer acquisition while reducing operational overhead.

---

## Project Scope

The MLOps pipeline automates the following stages:

1. Data ingestion and validation
2. Data cleaning and preprocessing
3. Feature engineering and transformation
4. Model training and evaluation
5. Model deployment readiness
6. CI/CD automation for continuous integration and updates

---

## Data Description

The dataset consists of **customer demographics and interaction attributes** used to predict the likelihood of purchasing the Wellness Tourism Package.

### Target Variable

* **ProdTaken**: Indicates whether the customer purchased the package

  * `0` – No
  * `1` – Yes

---

### Customer Details

* **CustomerID**: Unique identifier for each customer
* **Age**: Age of the customer
* **TypeofContact**: Mode of contact (Company Invited / Self Inquiry)
* **CityTier**: City category based on development and population (Tier 1, Tier 2, Tier 3)
* **Occupation**: Customer’s profession (Salaried, Freelancer, etc.)
* **Gender**: Gender of the customer
* **NumberOfPersonVisiting**: Total number of people traveling with the customer
* **PreferredPropertyStar**: Preferred hotel star rating
* **MaritalStatus**: Marital status (Single, Married, Divorced)
* **NumberOfTrips**: Average number of annual trips
* **Passport**: Passport availability (0: No, 1: Yes)
* **OwnCar**: Car ownership (0: No, 1: Yes)
* **NumberOfChildrenVisiting**: Number of children below 5 years
* **Designation**: Job designation
* **MonthlyIncome**: Monthly income of the customer

---

### Customer Interaction Data

* **PitchSatisfactionScore**: Customer satisfaction score for the sales pitch
* **ProductPitched**: Product pitched to the customer
* **NumberOfFollowups**: Number of follow-ups after the sales pitch
* **DurationOfPitch**: Duration (in minutes) of the sales pitch

---

## Technology Stack

* **Programming Language**: Python
* **Machine Learning**: Scikit-learn
* **Version Control**: Git & GitHub
* **CI/CD Automation**: GitHub Actions
* **MLOps Practices**: Automated training, testing, and deployment readiness

---

## Business Impact

This predictive MLOps solution enables:

* Accurate identification of high-potential customers
* Reduced manual effort and operational inefficiencies
* Improved marketing campaign performance
* Scalable and reproducible model deployment
* Faster adaptation to changing customer behavior

---

## Conclusion

By implementing an automated MLOps pipeline, **"Visit with Us"** can significantly enhance its marketing effectiveness, improve customer targeting accuracy, and drive sustainable business growth. This project demonstrates the practical application of MLOps principles to solve real-world business challenges in the tourism industry.
