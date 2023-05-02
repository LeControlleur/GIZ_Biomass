from preprocessing import BiomassPreprocesser
from catboost import CatBoostRegressor


if __name__=="__main__":

	BiomassPreprocesser.load_data()
	
	train_images = BiomassPreprocesser.train_images
	train_biomasses = BiomassPreprocesser.train_biomasses
	validate_images = BiomassPreprocesser.validate_images
	validate_biomasses = BiomassPreprocesser.validate_biomasses
	test_images = BiomassPreprocesser.test_images
	test_biomasses = BiomassPreprocesser.test_biomasses
	
	BiomassPreprocesser.features_selection()
	print(BiomassPreprocesser.MEAN.shape)
    
	print("Data transformation...")
	train_images_transformed = BiomassPreprocesser.data_transformer(
		train_images,
		use_selected_features=False
	)

	params = {'iterations': [50],
        #       'iterations': [50, 100, 200],
            'depth': [8, 15, 25],
            'l2_leaf_reg': [0.2, 0.8, 3]}
	
	print("Catboost regression application...")
	catboost_model = CatBoostRegressor(loss_function='RMSE')
    
	# Train the grid search model
	catboost_model.grid_search(param_grid=params, 
                    X=train_images_transformed, 
                    y=train_biomasses, 
                    verbose=True, 
                    plot=True)
	
	print("Training end, saving model ...")
	catboost_model.save_model("model_files/model_50", format="onnx")
    