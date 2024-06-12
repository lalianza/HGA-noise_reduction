import cv2
import numpy as np
from scipy.signal import wiener
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

class WienerFilter:
    def __init__(self):
        pass

    def wiener_filter(self, image, mysize=None, noise=None):
        filtered_image = wiener(image, mysize=mysize, noise=noise)
        return np.clip(filtered_image, 0, 255).astype(np.uint8)

    def calculate_mse(self, image1, image2):
        return mean_squared_error(image1.ravel(), image2.ravel())

    def find_best_parameters(self, image, param_grid):
        best_mse = float('inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            filtered_image = self.wiener_filter(image, **params)
            mse = self.calculate_mse(image, filtered_image)

            if mse < best_mse:
                best_mse = mse
                best_params = params

        return best_params, best_mse

    def smooth_image(self, image, param_grid=None):
        if param_grid is None:
            param_grid = {
                'mysize': [3, 5, 7, 9],
                'noise': [None, 0.01, 0.02, 0.05]
            }

        best_params, best_mse = self.find_best_parameters(image, param_grid)

        print(f"Melhores parâmetros: {best_params}")
        print(f"Melhor MSE: {best_mse}")

        # Apply Wiener filter with the best parameters
        filtered_image = self.wiener_filter(image, **best_params)

        return filtered_image


