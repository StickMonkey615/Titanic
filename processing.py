import pandas as pd
import numpy as np

# Conduct data pre-processing
def process_df(df, test=False):
    '''A function to conduct pre-processing tasks on a Pandas dataframe. Test variable indicates if the df being processed is the test split.'''
    processed_df = df

    def split_name(data):
        '''A function to split a string containing multiple words into a tokenised array of individual strings.'''
        return " ".join([v.strip(",()[].\"'") for v in data.split(" ")])

    def split_ticket_prefix(data):
        '''A function to split the ticket number string, returning only the string prefix and removing the integer numerical section.'''
        if data.split(" ")[0] == "LINE":
            print("HERE!")
        if len(data.split(" ")) == 1 and (data.split(" ")[0].isnumeric()):
            a = data.split(" ")[-1]
        elif len(data.split(" ")) == 1:
            a = "NIL"
        else:
            a = "_".join(data.split(" ")[0:-1])
        return a
    
    def split_ticket_no(data):
        '''A function to split the ticket number string, removing the string prefix and returning the integer numerical
        section.'''
        if data.split(" ")[0] == "LINE":
            print("HERE!")
        if len(data.split(" ")) == 1 and (data.split(" ")[0].isnumeric()):
            return data.split(" ")[-1]
        else:
            return 0
        
    def remove_nan(data):
        '''Remove all entries containing NaN and replace with 0.'''
        if np.isnan(data):
            # replace NaN with 0
            data=0
        return data
    
    def remove_features(data, features, test):
        '''A function to remove data features from a Pandas dataframe.'''
        if not test:
            data = data.drop(columns=features)
        return data
    
    def move_features(data, features, test):
        '''A function to move data features to the last column of a Pandas dataframe.'''
        if not test:
            print(list(data.columns))
            print(features)
            surv = data.pop('Survived')
            data.insert(len(data.columns), 'Survived', surv)
        return data
    
    def encode_cat_data(data):
        '''Temporarily remove all non-categorical data before encoding all categorical data (using dummy variables). On
        completion reinsert all non-catgorical data.'''
        # Remove non-categorical data
        age = data.pop('Age')
        fare = data.pop('Fare')
        ticket_no = data.pop('Ticket No')
        sibsp = data.pop('SibSp')
        parch = data.pop('Parch')

        # Encode the catagorical data (dummy variables)
        proc_data = pd.get_dummies(data=data, prefix_sep='_', drop_first=True)
    
        # Add back in non-categorical data
        proc_data.insert(0, 'Age', age)
        proc_data.insert(0, 'Fare', fare)
        proc_data.insert(0, 'Ticket No', ticket_no)
        proc_data.insert(0, 'SibSp', sibsp)
        proc_data.insert(0, 'Parch', parch)

        return proc_data

    processed_df["Name"] = df["Name"].apply(split_name)
    processed_df["Ticket prefix"] = df["Ticket"].apply(split_ticket_prefix)
    processed_df["Ticket No"] = df["Ticket"].apply(split_ticket_no)
    processed_df["Age"] = df["Age"].apply(remove_nan)
    processed_df = remove_features(processed_df, ["PassengerId", "Ticket"], test)
    processed_df = encode_cat_data(processed_df)
    processed_df = move_features(processed_df, ["Survived"], test)
    print(processed_df.head(10))
    return processed_df

def apply_scaling(df):
    '''A function to apply feature scaling to the passed Pandas dataframe.'''
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    processed_df = sc.fit_transform(df)
    del sc
    return processed_df
