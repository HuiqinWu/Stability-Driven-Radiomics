import numpy as np
import SimpleITK as sitk
import random
from my_utils.dice_3D import dice_coefficient
from my_utils.hausdorff_distance_3D import hausdorff_distance
from my_utils.iou_3D import intersection_over_union
import os
from decimal import Decimal




def read_image(img_path):
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img_array = sitk.GetArrayFromImage(img)    # z,y,x
    return img, img_array



def save_img(original_image, new_image, save_path):
    """
    此函数将数组按照original image的格式（位置信息和spacing）保存
    :param original_image:
    :param new_array:
    :return:
    """
    new_image.CopyInformation(original_image)
    sitk.WriteImage(new_image, save_path)


def findCenterSeries(maskArray):
    """

    :param maskArray: 3D Array
    :return: The average of cancer slices mask centers
    """
    slices_center_y = []
    slices_center_x = []

    for i in range(maskArray.shape[0]):
        coord_y, coord_x = np.nonzero(maskArray[i, :, :])

        # check if has the mask
        non_mask_flag = list(coord_y) == []
        if non_mask_flag:
            continue

        ymin = coord_y.min()
        ymax = coord_y.max()
        xmin = coord_x.min()
        xmax = coord_x.max()

        center_y = int((ymin+ymax) / 2)
        center_x = int((xmin+xmax) / 2)

        slices_center_y.append(center_y)
        slices_center_x.append(center_x)

    avg_center_y = int(np.mean(slices_center_y))
    avg_center_x = int(np.mean(slices_center_x))

    return avg_center_y, avg_center_x


def apply_random_affine_transform(image, image_arr, rotation_range=(-5, 5), translation_range=(-5, 5), scale_range=(0.95, 1.05),
                                  seed=None):
    """
    Apply a random affine transformation to a 3D image using SimpleITK.

    Parameters:
    image (sitk.Image): Input 3D binary image.
    rotation_range (tuple): Range of rotation angles (in degrees) for each axis.
    translation_range (tuple): Range of translations for each axis.
    scale_range (tuple): Range of scaling factors.
    shear_range (tuple): Range of shearing factors.

    Returns:
    sitk.Image: Transformed 3D image.
    """
    np.random.seed(seed)

    # Get the size and compute the center of the image
    size = image.GetSize()
    # center = [int(s/2) for s in size]

    center_y, center_x = findCenterSeries(image_arr)
    center_z = size[2] // 2
    center = [center_x, center_y, center_z]

    # Create an AffineTransform with dimensionality of 3 (for 3D)
    transform = sitk.AffineTransform(3)

    # Random rotation angles (in degrees) for each axis
    rotation_angles = np.random.uniform(rotation_range[0], rotation_range[1], size=3)
    rotation_radians = np.deg2rad(rotation_angles)

    # Compute rotation matrices for each axis
    cosx, cosy, cosz = np.cos(rotation_radians)
    sinx, siny, sinz = np.sin(rotation_radians)

    rotation_matrix = np.array([[cosy * cosz, -cosy * sinz, siny],
                                [sinx * siny * cosz + cosx * sinz, -sinx * siny * sinz + cosx * cosz, -sinx * cosy],
                                [-cosx * siny * cosz + sinx * sinz, cosx * siny * sinz + sinx * cosz, cosx * cosy]])

    # Set rotation component in the affine transform
    transform.SetMatrix(rotation_matrix.flatten())

    # Random scaling factors for each axis
    scaling_factors = np.random.uniform(scale_range[0], scale_range[1], size=3)
    scaling_matrix = np.diag(scaling_factors)

    # Combine scaling and rotation into the transformation matrix
    affine_matrix = np.dot(scaling_matrix, rotation_matrix)
    transform.SetMatrix(affine_matrix.flatten())

    # Random translations for each axis
    translation_factors = np.random.uniform(translation_range[0], translation_range[1], size=3)
    transform.SetTranslation(translation_factors.tolist())

    # Set the center of rotation to the center of the image
    transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint(center))

    # Resample the image using the affine transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)

    transformed_image = resampler.Execute(image)

    # Binarize the transformed image to maintain the mask structure
    transformed_image = sitk.BinaryThreshold(transformed_image, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1, outsideValue=0)

    return transformed_image


def generate_pseudo_mask(mask_array, mask_image, case_path, threshold_):
    """
    Generate a fake 3D medical image mask with random deformations.

    Returns:
    numpy.ndarray: 3D binary mask with continuous and deformed structures.
    """

    # 初始化随机种子
    count = 0

    # 循环执行，直到Dice系数超过阈值
    while count < 1000:
        # 如果没有种子，则生成一个新的随机种子
        seed = random.randint(0, 1000)
        print('Seed is {}'.format(seed))

         # 应用随机仿射变换
        pseudo_mask_image = apply_random_affine_transform(mask_image, mask_array, seed=seed)
        pseudo_mask_array = sitk.GetArrayFromImage(pseudo_mask_image)

        # 计算dice系数，hausdorff距离，iou
        dice_ = dice_coefficient(mask_array, pseudo_mask_array)
        hausdorff_distance_ = hausdorff_distance(mask_array, pseudo_mask_array)
        iou_ = intersection_over_union(mask_array, pseudo_mask_array)
        print('Dice is {}, hausdorff is {}, iou is {}'.format(dice_, hausdorff_distance_, iou_))

        if dice_ > threshold_ and iou_ > threshold_:
            save_path = case_path + 'mask_' + str(threshold_) + '_' + str(Decimal(dice_).quantize(Decimal("0.00"))) + '_' + \
                        str(Decimal(hausdorff_distance_).quantize(Decimal("0.00"))) + '_' \
                        + str(Decimal(iou_).quantize(Decimal("0.00"))) + '.nii.gz'
            save_img(mask_image, pseudo_mask_image, save_path)
            break

        count += 1
        print('****************** Count : {} ***************'.format(count))

    while count >= 1000 and count < 2000:
        # 如果没有种子，则生成一个新的随机种子
        seed = random.randint(0, 1000)
        print('Seed is {}'.format(seed))

         # 应用随机仿射变换
        pseudo_mask_image = apply_random_affine_transform(mask_image, mask_array,
                                                          rotation_range=(-3, 3),
                                                          translation_range=(-3, 3),
                                                          scale_range=(0.95, 1.05),
                                                          seed=seed)
        pseudo_mask_array = sitk.GetArrayFromImage(pseudo_mask_image)

        # 计算dice系数，hausdorff距离，iou
        dice_ = dice_coefficient(mask_array, pseudo_mask_array)
        hausdorff_distance_ = hausdorff_distance(mask_array, pseudo_mask_array)
        iou_ = intersection_over_union(mask_array, pseudo_mask_array)
        print('Dice is {}, hausdorff is {}, iou is {}'.format(dice_, hausdorff_distance_, iou_))

        if dice_ > threshold_ and iou_ > threshold_:
            save_path = case_path + 'mask_' + str(threshold_) + '_' + str(Decimal(dice_).quantize(Decimal("0.00"))) + '_' + \
                        str(Decimal(hausdorff_distance_).quantize(Decimal("0.00"))) + '_' \
                        + str(Decimal(iou_).quantize(Decimal("0.00"))) + '.nii.gz'
            save_img(mask_image, pseudo_mask_image, save_path)
            break

        count += 1
        print('****************** Count : {} ***************'.format(count))





# Example usage
if __name__ == "__main__":
    threshold_ = 0.6
    images_folder_path = 'D:/MVI_pseudo_mask/Data/Zhongshan'
    case_list = os.listdir(images_folder_path)
    for case in case_list:
        print('====================== Processing {} ============================'.format(case))
        case_path = images_folder_path + '/' + case + '/'
        mask_path = case_path + 'fill_mask.nii.gz'
        mask_image, mask_array = read_image(mask_path)
        image_num = len(os.listdir(case_path))
        generate_pseudo_mask(mask_array, mask_image, case_path, threshold_)
        print('Done {}'.format(case))

