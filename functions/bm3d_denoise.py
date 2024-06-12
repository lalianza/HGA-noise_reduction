import cv2
import numpy as np
import bm3d 
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.model_selection import KFold

class BM3DDenoiser:
    def __init__(self):
        pass

    def denoise_and_evaluate(self, image, sigma_psd):
        denoised_image = (bm3d.bm3d(image, sigma_psd=sigma_psd) * 255).astype(np.uint8)
        return denoised_image

    def cross_validate_sigma_psd(self, image, sigmas, n_splits=5):
        best_sigma = None
        best_psnr = -np.inf
        kf = KFold(n_splits=n_splits)

        for sigma in sigmas:
            psnr_scores = []
            for train_index, val_index in kf.split(image):
                train_image = image[train_index]
                val_image = image[val_index]
                denoised_image = self.denoise_and_evaluate(train_image, sigma)
                denoised_image_resized = cv2.resize(denoised_image, (val_image.shape[1], val_image.shape[0]))
                psnr_value = psnr(val_image, denoised_image_resized)
                psnr_scores.append(psnr_value)

            mean_psnr = np.mean(psnr_scores)

            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                best_sigma = sigma

        return best_sigma

    def bm3d_denoise(self, image, sigmas=None):
        if sigmas is None:
            sigmas = np.linspace(1/255, 200/255, 10)

        best_sigma = self.cross_validate_sigma_psd(image, sigmas)
        denoised_image = self.denoise_and_evaluate(image, best_sigma)
        return denoised_image


  