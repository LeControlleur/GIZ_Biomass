from xgboost import XGBRegressor
from preprocessing import BiomassPreprocesser
import numpy as np
from sklearn.model_selection import RandomizedSearchCV



if __name__=="__main__":

	BiomassPreprocesser.load_data()
	
	train_images = BiomassPreprocesser.train_images
	train_biomasses = BiomassPreprocesser.train_biomasses
	validate_images = BiomassPreprocesser.validate_images
	validate_biomasses = BiomassPreprocesser.validate_biomasses
	test_images = BiomassPreprocesser.test_images
	test_biomasses = BiomassPreprocesser.test_biomasses
	
	# BiomassPreprocesser.features_selection()
	# print(BiomassPreprocesser.MEAN.shape)
    
	print("Data transformation...")
	train_images_transformed = BiomassPreprocesser.data_transformer(
		train_images,
		use_selected_features=False)
	params = {
	    'max_depth': [6, 10, 15, 20, 25, 50],
	    'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': np.arange(0.5, 1.0, 0.1),
        'colsample_bytree': np.arange(0.4, 1.0, 0.1),
        'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
        'n_estimators': [100, 500, 1000, 5000, 10000]}
	
	xgboost_model = XGBRegressor(loss_function='RMSE')
	print("XGBoost regression application...")
	
	random_search = RandomizedSearchCV(
	    estimator=xgboost_model,
        param_distributions=params,
        scoring='neg_mean_squared_error',
        n_iter=25,
        n_jobs=-1,
        verbose=4
    )
    
	# Train the grid search model
	random_search.fit(
		X=train_images_transformed, 
        y=train_biomasses
	)
	

	print("Training end, saving model ...")
	print("Best parameters:", random_search.best_params_)
	print("Lowest RMSE: ", (-random_search.best_score_)**(1/2.0))

	random_search.best_estimator_.save_model("model_files/xgboosst_saved.json")
	random_search.best_estimator_.dump_model("model_files/xgboosst_dumped")
        

    