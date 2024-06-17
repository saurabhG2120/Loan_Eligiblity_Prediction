import os, sys
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_obj

import pandas as pd
import numpy as np


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class Datatransformation:
    def __init__(self) :
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_obj(self):
        try:
            numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
            categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
            ordinal_columns = ['Dependents', 'Credit_History']

            ordinal_pipe = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent",)),
                    ("ordinal_encode", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                    
                ]
            )

            cat_pipe = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy='most_frequent')),
                    ("onehot", OneHotEncoder( dtype=int,handle_unknown='ignore'))
                   
                ]
            )

            numeric_pipe = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                transformers=[
                    ("ordinal", ordinal_pipe, ordinal_columns),
                    ("categorical", cat_pipe, categorical_columns),
                    ("numerical", numeric_pipe, numerical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise Exception(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            preprocessing_obj=self.get_data_transformer_obj()
            target_column="Loan_Status"
            input_feature_train_df=train_df.drop(columns=["Loan_ID","Loan_Status"],axis=1)
            target_feature_train_df=train_df[target_column]


            input_feature_test_df=test_df.drop(columns=["Loan_ID","Loan_Status"],axis=1)
            target_feature_test_df=test_df[target_column]


            input_feature_train_preprocessed=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_preprocessed=preprocessing_obj.transform(input_feature_test_df)


            #Onehot encoding target variable

            encoder=OneHotEncoder(dtype=int,drop="first",sparse_output=False)
            target_feature_train_encoded=encoder.fit_transform(np.array(target_feature_train_df).reshape(-1,1)).ravel()
            target_feature_test_encoded=encoder.transform(np.array(target_feature_test_df).reshape(-1,1)).ravel()
        
            train_arr=np.c_[input_feature_train_preprocessed,target_feature_train_encoded]
            test_arr=np.c_[input_feature_test_preprocessed,target_feature_test_encoded]

            

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            #print(train_arr)
            return train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            raise Exception(e,sys)
            