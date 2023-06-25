import pandas as pd
import numpy as np
import csv
from classes.main import Main
from sklearn.model_selection import train_test_split

class Splitter(Main):
    def __init__(self):
        # Access to the Main class
        super().__init__()
        # If Splitter is active
        if self.cfg.splitter['active'] == True:
            # Call read_text 
            X_origin, y_origin = self.read_txt()
            X_aug, y_aug = self.read_txt2()

            # Split the text file
            X_train_origin, X_test_origin, y_train_origin, y_test_origin = self.split(X_origin, y_origin)
            # The 2nd split to create dev set using the new train set
            X_train_origin, X_dev_origin, y_train_origin, y_dev_origin = self.split(X_train_origin, y_train_origin)

            X_train_aug, X_test_aug, y_train_aug, y_test_aug = self.split(X_aug, y_aug)
            # The 2nd split to create dev set using the new train set
            X_train_aug, X_dev_aug, y_train_aug, y_dev_aug = self.split(X_train_aug, y_train_aug)


            X_train = np.concatenate((X_train_origin + X_train_aug), axis = 0)
            X_test = X_test_origin 
            X_dev = X_dev_origin + X_dev_aug
            y_train = np.concatenate((y_train_origin + y_train_aug))
            y_test = y_test_origin
            y_dev = y_dev_origin + y_dev_aug

            # Print out each split's length
            print("Train length:", len(X_train), len(y_train))
            print("Dev length:",   len(X_dev),   len(y_dev))
            print("Test length:",  len(X_test),  len(y_test))
            # Create dataframes from each array
            train = pd.DataFrame({'src': X_train, 'tgt':y_train})
            dev = pd.DataFrame({'src': X_dev, 'tgt':y_dev})
            test = pd.DataFrame({'src': X_test, 'tgt':y_test})
            # Print out the dataframes
            print(train)
            print(dev)
            print(test)
            # Save the dataframes into a csv file
            # Train
            train.to_csv(
                self.cfg.dataset['train_path'],
                quoting=csv.QUOTE_NONE,
                sep='\t', index=False)
            # Dev
            dev.to_csv(
                self.cfg.dataset['dev_path'],
                quoting=csv.QUOTE_NONE,
                sep='\t', index=False)
            # Test
            test.to_csv(
                self.cfg.dataset['test_path'],
                quoting=csv.QUOTE_NONE,
                sep='\t', index=False)

    def read_txt(self):
        """
        Reads the text file and load it in a dataframe
        Then returns the source and target of the text file
        """
        data = pd.read_csv(self.cfg.splitter['path'], 
                        header=None, quoting=csv.QUOTE_NONE,
                        sep='\t', names=['src', 'tgt']
                        ).dropna()
        print(data)
        X = data['src'].values
        y = data['tgt'].values
        return X, y


    def read_txt2(self):
        """
        Reads the text file and load it in a dataframe
        Then returns the source and target of the text file
        """
        data = pd.read_csv(self.cfg.splitter['path2'], 
                        header=None, quoting=csv.QUOTE_NONE,
                        sep='\t', names=['src', 'tgt']
                        ).dropna()
        print(data)
        X = data['src'].values
        y = data['tgt'].values
        return X, y    

    
    def split(self, X, y):
        """
        Take X and y and split them into two sets 
        Given the proportion of the dataset to include in the test/dev split
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.splitter['split'],
            shuffle=False)
        return X_train, X_test, y_train, y_test
    