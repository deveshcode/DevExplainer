import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from mosaicml import *
from mosaicml.constants import MLModelFlavours

def get_fifa_model():
    
    dataset = pd.read_csv("/data/fifa.csv")
#     X = dataset.drop(["short_name","nationality", "overall", "potential", "value_eur", "wage_eur"], axis = 1)
    X = dataset[['age', 'height_cm', 'weight_kg', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve']]
    y = dataset['value_eur']
    ylog = np.log(y)
    X_train, X_test, ylog_train, ylog_test, y_train, y_test = train_test_split(X, ylog, y, test_size=0.25, random_state=4)
    gbm_default = GradientBoostingRegressor()
    gbm_default.fit(X_train, y_train)
    
    return gbm_default, X_train, y_train
