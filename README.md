# Amruta XAI V2

Author: Vinh Nguyen, Co-Author: Dishit Kotecha

Extended version of V1 with modularized tab selection, more explainability features, and username and password management.

Built on Streamlit and Python

More information on developing with Streamlit: https://docs.streamlit.io/en/stable/

## About
Amruta XAI is a Python tool for Data Scientists, Data Science Enthusiasts to utilize various methods of AI Explainability for a wide variety of use cases. This tool provides an efficient and comprehensive user interface for easy modeling, evaluation, and explainability to save time on coding. It is built on popular dependencies such as Pandas, SKLearn, Numpy, etc. that are most common in a Data Scientist's toolbox, so it's not too unfamiliar to use. This app uses Streamlit as a front end framework. 

## Features

### Data Explorer
* View data
* View continuous attribute summary
* View Categorical attribute summary
* Column and Record count 
* Column list
* Correlation heatmap

### Data Processing
* Index/slice
* Drop columns
* Filter by column value
* Label Encoding
* Standard Scaling

### Modeling
* Binary Classification
* Multi-class Classification
* Regression
* Models
    - Linear/Logistic regression
    - Decision Tree
    - Random Forest
    - XGBoost
    - Light GBM
    - CatBoost

### Explainability
* Global Summary
* Local Explanation
* PDP/Interaction Plots
* Feature Importance

## Deployment

### Heroku deployment

Sign into https://dashboard.heroku.com/apps using your Amruta Inc. work email.

Create a new app with the button on the top right corner.

Enter app name.

Follow instructions under the 'Deploy' tab.


### To run locally

Navigate to directory
`cd .../amruta-xai-2`

Install requirements file
`pip install -r requirements.txt`

Run Streamlit App
`cd src`
`streamlit run app.py`

### To Run in Docker

Download Docker Desktop: https://www.docker.com/products/docker-desktop

And create a Docker account. You may need to login to Docker CLI in your Command Prompt/Terminal: docker login

Build image:
`docker build -t amruta_xai_2 .`

Push to repo:
`docker tag amruta_xai_2 <REPO_NAME>`
`docker push <REPO_NAME>`

If Docker image exists in a repo:

Pull from repo:
`docker pull <REPO_NAME>:amruta_xai_2`

Run in preferred browser (best in Chrome or Firefox)
`docker run -p 8501:8501 amruta_xai_2`

If that does not work, try:
`docker run -p 8501:8501 <REPO_NAME>`

Navigate to: `localhost:8501`

### To create new username and password

Show existing accounts:
`python src/account_management.py -show`

Add new user:
`python src/account_management.py -add`

Follow instructions via console

Delete a user:
`python src/account_management.py -delete`

Follow instructions via console
