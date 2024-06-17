import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.components.data_transformation import Datatransformation
from sklearn.model_selection import train_test_split
from src.components.model import ModelTrainer
import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv("data.csv")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_set,test_set=train_test_split(df,test_size=0.45,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise Exception(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=Datatransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))