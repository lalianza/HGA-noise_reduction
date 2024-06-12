import cv2
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

class AnisotropicDiffusion:
    def __init__(self):
        pass

    def anisotropic_diffusion(self, image, num_iterations, kappa, lambda_, option=1):
        diffused_image = image.astype(np.float32)

        for t in range(num_iterations):
            north = np.zeros_like(diffused_image)
            south = np.zeros_like(diffused_image)
            east = np.zeros_like(diffused_image)
            west = np.zeros_like(diffused_image)

            north[:-1, :] = diffused_image[1:, :] - diffused_image[:-1, :]
            south[1:, :] = diffused_image[:-1, :] - diffused_image[1:, :]
            east[:, :-1] = diffused_image[:, 1:] - diffused_image[:, :-1]
            west[:, 1:] = diffused_image[:, :-1] - diffused_image[:, 1:]

            if option == 1:
                c_north = np.exp(-(north/kappa)**2)
                c_south = np.exp(-(south/kappa)**2)
                c_east = np.exp(-(east/kappa)**2)
                c_west = np.exp(-(west/kappa)**2)
            elif option == 2:
                c_north = 1 / (1 + (north/kappa)**2)
                c_south = 1 / (1 + (south/kappa)**2)
                c_east = 1 / (1 + (east/kappa)**2)
                c_west = 1 / (1 + (west/kappa)**2)

            diffused_image += lambda_ * (c_north * north + c_south * south + c_east * east + c_west * west)

        return diffused_image.astype(np.uint8)

    def calculate_mse(self, image1, image2):
        return mean_squared_error(image1.ravel(), image2.ravel())

    def find_best_parameters(self, image, param_grid):
        best_mse = float('inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            diffused_image = self.anisotropic_diffusion(image, **params)
            mse = self.calculate_mse(image, diffused_image)

            if mse < best_mse:
                best_mse = mse
                best_params = params

        return best_params, best_mse

    def smooth_image(self, image, param_grid=None):
        if param_grid is None:
            param_grid = {
                'num_iterations': [5, 10, 15],
                'kappa': [20, 50, 100],
                'lambda_': [0.1, 0.15, 0.2, 0.25],
                'option': [1, 2]
            }

        best_params, best_mse = self.find_best_parameters(image, param_grid)

        print(f"Melhores parâmetros: {best_params}")
        print(f"Melhor MSE: {best_mse}")

        # Apply anisotropic diffusion with the best parameters
        diffused_image = self.anisotropic_diffusion(image, **best_params)

        return diffused_image