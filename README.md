# Predicting Solar Energy Potential with Machine Learning

## Introduction

This project aims to predict the solar energy potential using a Random Forest Regressor model. Unlike traditional methods, this approach attempts to forecast solar power output without relying on irradiance data, which is typically a crucial factor in predicting solar output. The model achieves an R squared score of 0.67, indicating its effectiveness in capturing the variance of the dataset.

## Background

The data for this project was sourced from the research paper "Machine Learning Modeling of Horizontal Photovoltaics Using Weather and Location Data" by Christil Pasion, Torrey Wagner, Clay Koschnick, Steven Schuldt, Jada Williams, and Kevin Hallinan. This study provided a comprehensive dataset collected from 12 sites over a span of 14 months, which was instrumental in developing our predictive model.

## Project Highlights

- **Model Development**: Developed a Random Forest Regressor model that significantly predicts solar energy potential with an R squared score of 0.68.
- **Data Analysis**: Performed Exploratory Data Analysis over the data and found important features for model building.
- **Deployment**: The project includes a complete pipeline for deployment on platforms like AWS, Azure, etc.
- **Technology Stack**: Utilized technologies such as Pandas for data manipulation, Matplotlib for data visualization, Scikit-learn for machine learning modeling, and Flask for creating the back-end. Additionally, integrated a CI-CD pipeline to streamline the development and deployment process.

