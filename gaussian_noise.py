import SimpleITK as sitk
import numpy as np
import os


def add_gaussian_noise(image, mean=0, std=1):
    """
    Add Gaussian noise to a 3D SimpleITK image.

    Parameters:
    - image: SimpleITK Image containing 3D data.
    - mean: Mean of the Gaussian noise.
    - std: Standard deviation of the Gaussian noise.

    Returns:
    - noisy_image: SimpleITK Image with added Gaussian noise.
    """
    # Convert SimpleITK image to NumPy array
    image_array = sitk.GetArrayFromImage(image)

    # Add Gaussian noise to the NumPy array
    noise = np.random.normal(mean, std, image_array.shape)
    noisy_array = image_array + noise

    # Clip the values to ensure they remain within the valid range
    # Replace `min_val` and `max_val` with appropriate intensity limits for your image.
    min_val = image_array.min()
    max_val = image_array.max()
    noisy_image_array = np.clip(noisy_array, min_val, max_val)

    # Convert the noisy NumPy array back to SimpleITK image
    noisy_image = sitk.GetImageFromArray(noisy_array)

    # Copy the original image's metadata to the noisy image
    noisy_image.CopyInformation(image)

    return noisy_image

if __name__ == '__main__':

    sigma_list = [5]
    for sigma in sigma_list:

        # Read the 3D medical image
        image_folder_path = 'I:/MVI_pseudo_mask/Data/Hunan/'
        case_list = os.listdir(image_folder_path)
        for case in case_list:
            case_path = image_folder_path + case + '/'

            input_image_path_A = case_path + 'A.nii.gz'
            input_image_path_P = case_path + 'P.nii.gz'
            input_image_path_Gd = case_path + 'Gd.nii.gz'

            output_image_path_A = case_path + 'A_noise_sigma' + str(sigma) + '.nii.gz'
            output_image_path_P = case_path + 'P_noise_sigma' + str(sigma) + '.nii.gz'
            output_image_path_Gd = case_path + 'Gd_noise_sigma' + str(sigma) + '.nii.gz'

            if not os.path.exists(output_image_path_A):
                image_A = sitk.ReadImage(input_image_path_A)
                noisy_image_A = add_gaussian_noise(image_A, mean=0, std=sigma)
                sitk.WriteImage(noisy_image_A, output_image_path_A)
                print('Preprocessing A image')

            if not os.path.exists(output_image_path_P):
                image_P = sitk.ReadImage(input_image_path_P)
                noisy_image_P = add_gaussian_noise(image_P, mean=0, std=sigma)
                sitk.WriteImage(noisy_image_P, output_image_path_P)
                print('Preprocessing P image')

            if not os.path.exists(output_image_path_Gd):
                image_Gd = sitk.ReadImage(input_image_path_Gd)
                noisy_image_Gd = add_gaussian_noise(image_Gd, mean=0, std=sigma)
                sitk.WriteImage(noisy_image_Gd, output_image_path_Gd)
                print('Preprocessing Gd image')

            print('Done {}'.format(case))
