# Alphabet-Soup-Funding-Prediction

Background

Alphabet Soup, a nonprofit foundation, seeks a tool to help select applicants for funding with the best chance of success. Using machine learning and neural networks, this project aims to create a binary classifier to predict the success of applicants based on features in a provided dataset.

Dataset
The dataset contains information on more than 34,000 organizations that have received funding from Alphabet Soup, with columns including:

EIN and NAME — Identification columns
APPLICATION_TYPE — Alphabet Soup application type
AFFILIATION — Affiliated sector of industry
CLASSIFICATION — Government organization classification
USE_CASE — Use case for funding
ORGANIZATION — Organization type
STATUS — Active status
INCOME_AMT — Income classification
SPECIAL_CONSIDERATIONS — Special considerations for application
ASK_AMT — Funding amount requested
IS_SUCCESSFUL — Whether the money was used effectively

Project Steps

Step 1: Preprocess the Data

1. Data Loading and Initial Exploration:

Loaded the dataset into a Pandas DataFrame.
Identified IS_SUCCESSFUL as the target variable and the remaining relevant columns as features.
Dropped the EIN and NAME columns as they were not useful for prediction.

2. Handling Unique Values:

Analyzed the unique values in each categorical feature.
For columns with many unique values, combined infrequent categories into a new category labeled "Other" to reduce dimensionality.

3. Encoding Categorical Variables:

Used pd.get_dummies() to perform one-hot encoding on categorical variables.

4. Data Splitting:

Split the data into features (X) and target (y).
Further divided the data into training and testing sets using train_test_split.

5. Feature Scaling:

Applied StandardScaler to standardize the feature data for both training and testing sets.

Step 2: Compile, Train, and Evaluate the Model

1. Model Design:

Designed a neural network model using TensorFlow and Keras.
Created hidden layers with appropriate activation functions.
Included an output layer for binary classification with a sigmoid activation function.

2. Model Compilation and Training:

Compiled the model with binary cross-entropy loss and an Adam optimizer.
Trained the model, incorporating a callback to save the model's weights every five epochs.

3. Model Evaluation:

Evaluated the model on the test data to determine loss and accuracy.
Saved the final trained model to an HDF5 file named AlphabetSoupCharity.h5.

Step 3: Optimize the Model

1. Model Optimization:

Made several attempts to improve model performance, including:
Adjusting the neural network architecture by adding more neurons and layers.
Experimenting with different activation functions.
Tweaking the training process by altering the number of epochs.
Achieved a target accuracy higher than 75%.

2. Optimization Results:

Saved the optimized model to an HDF5 file named AlphabetSoupCharity_Optimization.h5.

Step 4: Reporting

1. Data Preprocessing:

Identified IS_SUCCESSFUL as the target variable.
Used all other columns except EIN and NAME as features.

2. Model Development:

Created a neural network with multiple layers and experimented with different architectures to optimize performance.
Evaluated the model's performance and discussed steps taken to achieve the desired accuracy.

