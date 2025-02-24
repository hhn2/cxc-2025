# PromoPulse - cxc2025

### Table of Contents
* [Introduction](#Introduction)
* [Getting Started](#Getting-Started)


## Introduction
PromoPulase takes in the user's csv file with their restaurant's past data and we will recommend when to give promotions the following week, based on our two machine learning models: earnings prediction model and potential earnings prediction. 


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