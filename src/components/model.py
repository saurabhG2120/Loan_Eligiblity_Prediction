from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import os,sys
import warnings
warnings.filterwarnings("ignore")
from src.utils import  evaluate_model,save_obj
from dataclasses import dataclass

@dataclass
class ModelConfig:
        train_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
        def __init__(self):
                self.modeltrainer=ModelConfig()


        def initiate_model_trainer(self,train_arr,test_arr):
                try:
                        X_train,y_train,X_test,y_test=(train_arr[:,:-1],
                                                       train_arr[:,-1],
                                                       test_arr[:,:-1],
                                                       test_arr[:,-1]
                                                       )
                        models={
                        "DecisionTree":DecisionTreeClassifier(),
                        'LogisticRegression': LogisticRegression(),
                        "SVM":SVC(),
                        "RandomForest":RandomForestClassifier(),
                        "GradientBoost":GradientBoostingClassifier(),
                        "GaussianNB":GaussianNB()
                        }
                        params={
                        "DecisionTree":{
                                'criterion':["gini", "entropy", "log_loss"],
                                'splitter':["best","random"],
                                'max_depth':[10,30,50,90,100,150,170,200]
                        },
                        'LogisticRegression': {
                                'penalty': ['l1', 'l2'],
                                'solver': ['liblinear', 'saga'],  # Both 'liblinear' and 'saga' support 'l1'
                                'C': [0.1, 1, 10]
                        },
                        
                        'SVM':{
                                'C':[0.1,0.8,0.9,1,2,5,7,10],
                                'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                                'gamma':["scale","auto"]
                                
                        },
                        "RandomForest":{
                                'criterion':["gini", "entropy", "log_loss"],
                                'max_depth':[10,20,25,30,35,40,45,50],
                                
                        },
                        "GradientBoost":{
                                'loss':['log_loss','exponential'],
                                'learning_rate':[0.1,0.5,0.8,1.0],
                                'criterion':['friedman_mse','squared_error']
                                
                        },
                        'GaussianNB':{
                                'var_smoothing': np.logspace(-12, -6, num=7)
                        }
                        }

                        model_report=evaluate_model(X_train=X_train,y_train=y_train,
                                                    X_test=X_test,y_test=y_test,
                                                    models=models,parameters=params)
                        best_model_name=max(model_report,key=model_report.get)
                        best_model=models[best_model_name]
                        
                        save_obj(file_path=self.modeltrainer.train_model_file_path,
                                 obj=best_model)
                        
                         
                        predicted=best_model.predict(X_test)
                        accuracy=accuracy_score(y_test,predicted)
                        #print(best_model,accuracy)
                        
                        return accuracy,best_model,model_report,predicted,
                except Exception as e:
                        raise Exception(e,sys)