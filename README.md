# Disaster Response Pipeline Project
![screenshot](https://github.com/DiloromA/Disaster_Response/blob/main/media/screenshot_1.JPG?raw=true)
### Overview
This is a project that uses natural language processing (NLP) to classify text in to relavant categories. The project goal is to help reduce the reaction time of disaster response teams by quickly identifying scale of disaster, volume of requests for aid and etc by classifing incoming texts in disaster situations. The project could be adapted or/optimized to use in similar situations. 

Two data sets that contain text messages and categories were provided by FigureEight via Udacity Data Scientist Nano Degree program. Text contains  messages in different languages (with translation into English) that were sent during disasters. In this project, I used NLP and Machine Learning Pipeline to classify messages in to one or more of 36 categories. 

### Project Structure
1. The ETL pipeline-located at Data directory

    * Combines the two datasets 

    * Cleans the data

    * Stores them in a SQLite database

2. The NLP and Machine Learning pipeline - located at Models diretory

    * Splits the dataset into training and test sets

    * Builds a text processing and machine learning pipeline

    * Trains and tunes a model using GridSearchCV

    * Outputs results on the test set

    * Exports the trained model as a pickle file

3. The Flask app -located at app direcotry
A web app that displays the model results. A user can enter a message to classify into one of 36 categories. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements
1. GitHub users jkarakas and xiaoye318024.
