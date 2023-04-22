# Diabetes Classification Web App with Docker
This is a simple web application built with Streamlit that allows the user to upload a dataset in CSV format and perform two different machine learning tasks on it: Classification and Clustering. This project has been Dockerized for easy deployment and scalability.

Project Description
This project was developed as part of a university project to explore the application of machine learning in healthcare. The goal was to build a web application that could classify a given dataset of medical records as diabetic or non-diabetic and also perform clustering to identify patterns in the data.

# Requirements
To run this application using Docker, you need to have Docker and Docker Compose installed on your system. You can download Docker here and Docker Compose here.

# How to use
To use this application, follow the steps below:

* Clone the repository to your local machine
* In the project directory, run `docker-compose up`
* Open a web browser and navigate to localhost:8501
# Classification
After uploading your dataset, you will be able to select the target column and perform a Random Forest classification task. By default, the number of estimators is set to 500 and the maximum depth to 10, but you can adjust these parameters using the sliders provided. You can also choose to perform 5-fold cross-validation or set a specific training set size.

# Clustering
In addition to classification, you can also perform K-Means clustering on your dataset. You can select the number of clusters using the slider provided. Once you run the clustering task, the application will display the Silhouette Score, Calinski-Harabasz Index, and the cluster centers.

# Dockerfile
The Dockerfile is used to build the Docker image for the application. The Dockerfile includes all the necessary dependencies to run the application.

# docker-compose.yml
The docker-compose.yml file is used to define the Docker services for the application. The file includes the service for the web application as well as the volume for the data.

