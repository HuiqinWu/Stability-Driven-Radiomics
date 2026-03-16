import SimpleITK as sitk
import os


def gaussian_smoothing_3d(input_image_path, output_image_path, sigma):
    """
    对3D医学图像进行高斯滤波平滑处理，并保存处理后的图像。

    参数:
    input_image_path (str): 输入图像的路径。
    output_image_path (str): 输出图像的路径。
    sigma (float or tuple of floats): 高斯滤波器的标准差。如果是浮点数，则所有方向使用相同的标准差；
                                     如果是元组，则每个方向可以使用不同的标准差。

    返回:
    None
    """
    # 读取3D医学图像
    image = sitk.ReadImage(input_image_path)

    # 创建高斯滤波器
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, sigma=sigma)
    smoothed_image.CopyInformation(image)

    # 保存处理后的图像
    sitk.WriteImage(smoothed_image, output_image_path)



if __name__ == '__main__':

    sigma_list = [1.5]
    for sigma in sigma_list:

        # Read the 3D medical image
        image_folder_path = 'I:/MVI_pseudo_mask/Data/Hunan/'
        case_list = os.listdir(image_folder_path)
        for case in case_list:
            case_path = image_folder_path + case + '/'
            input_image_path_A = case_path + 'A.nii.gz'
            input_image_path_P = case_path + 'P.nii.gz'
            input_image_path_Gd = case_path + 'Gd.nii.gz'

            # Save the smooth image to disk
            # output_image_path_A = case_path + 'A_smooth_sigma' + str(sigma) + '.nii.gz'
            output_image_path_A = case_path + 'A_smooth_sigma1p5.nii.gz'
            # output_image_path_P = case_path + 'P_smooth_sigma' + str(sigma) + '.nii.gz'
            output_image_path_P = case_path + 'P_smooth_sigma1p5.nii.gz'
            # output_image_path_Gd = case_path + 'Gd_smooth_sigma' + str(sigma) + '.nii.gz'
            output_image_path_Gd = case_path + 'Gd_smooth_sigma1p5.nii.gz'

            gaussian_smoothing_3d(input_image_path_A, output_image_path_A,
                                                   sigma=sigma)
            gaussian_smoothing_3d(input_image_path_P, output_image_path_P,
                                                   sigma=sigma)
            gaussian_smoothing_3d(input_image_path_Gd, output_image_path_Gd,
                                                    sigma=sigma)


            print('Done {}'.format(case))

