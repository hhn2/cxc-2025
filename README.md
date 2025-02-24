# PromoPulse - cxc2025

### Table of Contents
* [Introduction](#Introduction)
* [Getting Started](#Getting-Started)

## Inspiration
As UberEats users, we always look for the most cost-effective way to order, knowing that food delivery is typically more expensive than dining in. This is why promotions are our top priority when ordering. While this reflects the consumer perspective, we became curious about how restaurants view promotions. After all, promotions wouldn’t exist if they weren’t also beneficial to the restaurant. We assumed that restaurants would only offer discounts when they help increase overall earnings. However, determining whether a promotion actually leads to higher profits can be challenging.

## What it does
The web app takes in the user(restaurant)'s csv file with their restaurant's past POS data and recommends when to give promotions the following week, based on our two machine learning models, earnings prediction model and potential earnings prediction, and the predicted weather of the restaurant location.

## How we built it
During the data preprocessing stage, we removed all venues with pop-up and fine dining concepts, as these restaurants were less likely to be influenced by promotions or short-term market fluctuations. Additionally, we dropped columns with indistinguishable null values to ensure cleaner, more reliable input data.

The Earnings Prediction Model was built using XGBoost, a machine learning model well-suited for structured time-series data. Recognizing that recent data carries more importance due to economic shifts, seasonality, and changing trends, we applied time-based weighting during model training. This gave higher importance to more recent observations while still leveraging historical data for trend analysis. The model was trained on hourly aggregated revenue data, incorporating key features such as bill totals, payment amounts, time-based factors, and venue-specific attributes to predict future earnings accurately.

The Potential Earnings Prediction Model was also built using XGBoost, leveraging historical data to estimate the best possible earnings per hour. Instead of predicting expected revenue, this model identifies peak historical earnings by analyzing past performance trends and benchmarking the 90th percentile revenue for similar conditions. To ensure accuracy, we incorporated key features such as bill totals, payment amounts, order volume, time-based patterns, and venue characteristics. Like the Earnings Prediction Model, we applied time-based weighting, prioritizing more recent data to reflect evolving customer behaviour and market trends. This allowed the model to dynamically adjust to seasonal demand shifts and external influences. 

By comparing predicted potential earnings with actual earnings, restaurants can pinpoint underperforming hours, uncover revenue gaps, and optimize strategies to maximize profitability. And we decided to make these times a 'promotion-giving time'. 

## Challenges we ran into
The greatest challenge was getting the free Weather API for the large data we have. With millions of data points, it was hard to make API calls for the weather at the specific time. However, we were able to generalize the weather by using daily weather instead of hourly weather at a certain city. 

## Accomplishments that we're proud of
We successfully built a machine learning model that can analyze restaurant data and provide valuable insights for promotion timing. Our XGBoost model still delivers reliable predictions by prioritizing recent data patterns. We're also proud of developing an intuitive web interface that makes complex prediction models accessible to restaurant owners without requiring technical expertise.

## What we learned
We gained valuable experience in time-series data analysis and the implementation of weighted training in XGBoost models. We learned how to preprocess restaurant data effectively by identifying and removing outliers.

## What's next for PromoPulse
Our next step is expanding PromoPulse to help future restaurant owners by incorporating location-based sales estimation. This feature would allow entrepreneurs to predict potential earnings at different locations before opening, helping them make informed decisions about restaurant placement. We also plan to integrate hourly weather APIs for more accurate predictions and develop additional models that can suggest optimal promotion types.

## Getting Started
To get started with this project, you'll need to clone the repository and set up a virtual environment. This will allow you to install the required dependencies without affecting your system-wide Python installation.

### Cloning the Repository

    git clone https://github.com/hhn2/cxc-2025.git

### Setting up a Virtual Environment

    cd ./cxc-2025

    pyenv versions

    pyenv local 3.11.6

    echo '.env'  >> .gitignore
    echo '.venv' >> .gitignore

    python -m venv .venv        # create a new virtual environment

    source .venv/bin/activate   # Activate the virtual environment

### Install the required dependencies

    pip3 install -r requirements.txt

### Running the Application

    python3 -m streamlit run app.py
    
### Deactivate the virtual environment

    deactivate


# Images



## Developer Team

- [Mathilda Lee](https://github.com/jkmathilda)  
- [Hannah Hwang](https://github.com/hhn2)
