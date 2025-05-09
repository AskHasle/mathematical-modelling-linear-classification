from visualization_utils import visualize_unsigned_orientations_max
from model_utils import load_model

model = load_model('/Users/askhasle/Documents/GitHub/mathematical-modelling-linear-classification?fbclid=IwZXh0bgNhZW0CMTEAAR4RscCjxDMDP6JY_FFo9OmX-6zw3rEH5aM9SfZn8vUvl7zQZEwZgmj3tH9Pxg_aem__KVEWNgvSE-K9QRLML9l3Q/models/part_3_logistic_regression_model_20250429_105329.joblib')



visualize_unsigned_orientations_max(model, image_shape=(224, 224), pixels_per_cell=(32, 32), orientations=9)