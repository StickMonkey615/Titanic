'''Use Machine Learning algorithms to predict which passengers on the Titanic sinking based on individual passenger information. Based on the Kaggle challenge at https://www.kaggle.com/competitions/titanic.'''

# Import dependencies
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Import datasets
train_df = pd.read_csv("train.csv") 
test_df = pd.read_csv("test.csv")

# Conduct data pre-processing
def process_df(df, test=False):
    '''A function to conduct pre-processing tasks on a Pandas dataframe. Test variable indicates if the df being processed is the test split.'''
    processed_df = df

    def split_name(data):
        '''A function to split a string containing multiple words into a tokenised array of individual strings.'''
        return " ".join([v.strip(",()[].\"'") for v in data.split(" ")])

    def split_ticket_prefix(data):
        '''A function to split the ticket number string, returning only the string prefix and removing the integer numerical section.'''
        if len(data.split(" ")) == 1:
            a = "NIL"
        else:
            a = "_".join(data.split(" ")[0:-1])
        return a
    
    def split_ticket_no(data):
        '''A function to split the ticket number string, removing the string prefix and returning the integer numerical section.'''
        return data.split(" ")[-1]
    
    def remove_features(data, features, test):
        '''A function to remove data features from a Pandas dataframe.'''
        if not test:
            data = data.drop(columns=features)
        return data

    processed_df["Name"] = df["Name"].apply(split_name)
    processed_df["Ticket prefix"] = df["Ticket"].apply(split_ticket_prefix)
    processed_df["Ticket No"] = df["Ticket"].apply(split_ticket_no)
    processed_df = remove_features(processed_df, ["PassengerId", "Survived", "Ticket"], test)
    return processed_df

processed_train_df = process_df(train_df)
input_features = list(processed_train_df.columns)
print(f"Input features: {input_features}")
processed_test_df = process_df(test_df, test=True)
print(processed_train_df.head(10))

# Convert Pandas DataFrame to Tensorflow dataset
def tokenise_names(features, labels=None):
    '''Divide the names into tokens. F-DF can consume text tokens natively.'''
    features["Name"] = tf.strings.split(features["Name"])
    return features, labels

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(processed_train_df, label="Survived").map(tokenise_names)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(processed_test_df).map(tokenise_names)

# Train model
model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0,
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True, # Only use the features in "features"
    random_seed=1234,
)
model.fit(train_ds)

self_evaluation = model.make_inspector().evaluation()
print(f"Accuracy: {self_evaluation.accuracy} Loss: {self_evaluation.loss}")