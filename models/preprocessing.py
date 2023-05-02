import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression,Ridge,Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class CustomScaler(BaseEstimator, TransformerMixin):
  def __init__(self,mean,std):
    self.mean = mean
    self.std = std
  
  def fit(self, X, y=None):
    #self.mean = X.mean((0,2,3)) 
    #self.std = X.std((0,2,3))
    return self

  def transform(self, X, y=None):
    return (X-self.mean[None,:,None,None])/self.std[None,:,None,None] 

  def reverse_transform(self):
    return None


class FlattenTransformer(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    return X.reshape((X.shape[0], -1))


class BiomassPreprocesser:

    train_images : np.array = None
    train_biomasses : np.array = None
    validate_images : np.array = None
    validate_biomasses : np.array = None
    test_images : np.array = None
    test_biomasses : np.array = None
    selected_features : np.array = None
    MEAN = None
    STD = None


    def load_train_data() -> tuple[np.array, np.array]:
        print("Train data loading ....")
        trainset = h5py.File("/content/drive/MyDrive/vscode-ssh/GIZ_Biomass/data/09072022_1154_train.h5", "r")
        train_images = np.array(trainset['images'],dtype=np.float64)
        train_images = train_images.transpose(0,3,1,2)
        train_biomasses = np.array(trainset['agbd'],dtype=np.float64)

        BiomassPreprocesser.train_images = train_images
        BiomassPreprocesser.train_biomasses = train_biomasses

        return train_images, train_biomasses


    def load_validate_data() -> tuple[np.array, np.array]:

        print("Validation data loading ....")
        validateset = h5py.File("/content/drive/MyDrive/vscode-ssh/GIZ_Biomass/data/09072022_1154_val.h5", "r")
        validate_images = np.array(validateset['images'],dtype=np.float64)
        validate_images = validate_images.transpose(0,3,1,2)
        validate_biomasses = np.array(validateset['agbd'],dtype=np.float64)

        BiomassPreprocesser.validate_images = validate_images
        BiomassPreprocesser.validate_biomasses = validate_biomasses

        return validate_images, validate_biomasses


    def load_test_data() -> tuple[np.array, np.array]:

        print("Test data loading ....")
        testset = h5py.File("/content/drive/MyDrive/vscode-ssh/GIZ_Biomass/data/09072022_1154_test.h5", "r")
        test_images = np.array(testset['images'],dtype=np.float32)
        test_images = test_images.transpose(0,3,1,2)
        test_biomasses = np.array(testset['agbd'],dtype=np.float32)

        BiomassPreprocesser.test_images = test_images
        BiomassPreprocesser.test_biomasses = test_biomasses

        return test_images, test_biomasses


    def load_data() -> None:
        BiomassPreprocesser.load_train_data()
        BiomassPreprocesser.load_validate_data()
        BiomassPreprocesser.load_test_data()


    def features_selection() -> np.array:

        train_images = BiomassPreprocesser.train_images
        train_biomasses = BiomassPreprocesser.train_biomasses
        selected_features = BiomassPreprocesser.selected_features

        # constante for standardization
        BiomassPreprocesser.MEAN = train_images.mean((0,2,3))
        BiomassPreprocesser.STD = train_images.std((0,2,3))

        #   Params definitions
        # params_lasso = {
        #     "classifier__alpha": [1e-1],
        #     "classifier" : [Lasso(tol=1e-2)]
        # }

        # pipe = Pipeline(steps=[("scaler", CustomScaler(BiomassPreprocesser.MEAN, BiomassPreprocesser.STD)),
        #                     ("flatten", FlattenTransformer()),
        #                     ("classifier", Lasso())])

        # # Train the grid search model
        # grid_search = GridSearchCV(pipe, params_lasso, cv=3, scoring='neg_mean_squared_error', verbose=3).fit(train_images,train_biomasses)

        # coefficients = grid_search.best_estimator_.named_steps['classifier'].coef_

        pipe = Pipeline(steps=[("scaler", CustomScaler(BiomassPreprocesser.MEAN, BiomassPreprocesser.STD)),
                            ("flatten", FlattenTransformer()),
                            ("classifier", Lasso(alpha=1e-1, tol=1e-2))])

        # Train the pipeline
        pipe.fit(train_images,train_biomasses)
        coefficients = pipe['classifier'].coef_


        features = range(coefficients.shape[0])
        importance = np.abs(coefficients)
        selected_features = np.array(features)[importance > 0]
        BiomassPreprocesser.selected_features = selected_features

        return selected_features


    def data_transformer(X : np.array, 
                        standardize : bool = True, 
                        compute_standardize_variables : bool = True, 
                        selected_features : np.array = None,
                        use_selected_features : bool = True) -> np.array : 

        if selected_features is None:
            selected_features = BiomassPreprocesser.selected_features
        
        flatten_layer = lambda X : X.reshape((X.shape[0], -1))
        features_selection_layer = lambda X, features : np.array(X[:, features])

        images_flattened = flatten_layer(X)
        images_resized = features_selection_layer(images_flattened, selected_features) if use_selected_features else images_flattened
        if standardize :
            if compute_standardize_variables:
                BiomassPreprocesser.MEAN = np.mean(images_resized, axis=0)
            if compute_standardize_variables:
                BiomassPreprocesser.STD = np.std(images_resized, axis=0)
            standardization_layer = lambda X : (X-BiomassPreprocesser.MEAN)/BiomassPreprocesser.STD
            standardized_data = standardization_layer(images_resized)
        else :
            standardized_data = images_resized 
        return standardized_data
