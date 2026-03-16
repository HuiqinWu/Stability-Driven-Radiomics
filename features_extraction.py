import re

import pandas as pd
from radiomics import featureextractor
import os


def re_find_in_dir(path: str = '', pattern: list = []):
    """
    在指定目录下，查找符合规则的目录、文件。规则有多个时，拼接成 '*a*b' 进行匹配
    :param path: 指定目录
    :param pattern: 匹配规则
    :return: 符合规则的结果
    """
    match_file = []
    pattern_str = '.*' + '.*'.join(pattern)
    re_pattern = re.compile(pattern=pattern_str)

    file_list = os.listdir(path)
    for file_name in file_list:
        if re_pattern.search(file_name):
            match_file.append(file_name)

    return match_file


def extract_features(images_path, mask_name, modality, paramsFile, save_path_):
    """

    :param images_path:
    :param modality: T1C | T2F
    :param paramsFile:
    :param save_name:
    :return:
    """
    # Initialize feature extractor using the settings file
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    columns_names = []

    case_list = os.listdir(images_path)
    case_list.sort()

    patient_count = 0

    for case in case_list:
        case_path = images_path + case + '/'
        img_path = case_path + modality + '.nii.gz'
        mask_path = case_path + mask_name

        featureDict = extractor.execute(img_path, mask_path)

        if len(columns_names) == 0:              # empty list for the first time
            for name in featureDict.keys():
                if 'diagnostics' not in name:
                    columns_names.append(name)
            columns_names.sort()
            columns_names.insert(0, 'ID')
            features_df = pd.DataFrame(columns=columns_names)

        for feature_name in columns_names[1:]:       # skip 'ID'
            features_df.loc[patient_count, feature_name] = featureDict[feature_name]
        features_df.loc[patient_count, 'ID'] = case

        patient_count += 1
        print('Done {}'.format(case))

    features_df.to_csv(save_path_, index=False)








if __name__ == '__main__':
    images_path = 'I:/MVI_pseudo_mask/Data/Zhongshan/'
    modality_list = ['A']
    mask_name_list = [
                      'fill_mask.nii.gz',
                      'fill_mask_RegionDilate-1.00mm.nii.gz',
                      'fill_mask_RegionDilate-2.00mm.nii.gz',
                      'fill_mask_RegionDilate-3.00mm.nii.gz',
                      'fill_mask_RegionDilate-4.00mm.nii.gz',
                      'fill_mask_RegionDilate-5.00mm.nii.gz',
                      'fill_mask_RegionErode-1.00mm.nii.gz',
                      'fill_mask_RegionErode-2.00mm.nii.gz',
                      'fill_mask_RegionErode-3.00mm.nii.gz',
                      'mask_0.6.nii.gz',
                      'mask_0.7.nii.gz',
                      'mask_0.8.nii.gz',
                      ]

    for modality in modality_list:
        params_file = 'test.yaml'
        save_path = 'features/Zhongshan/' + modality + '/'
        os.makedirs(save_path, exist_ok=True)

        for mask_name in mask_name_list:
            # save_feat_path = save_path + mask_name[:-7] + '.cvs'
            save_feat_path = save_path + 'test.cvs'
            extract_features(images_path, mask_name, modality, params_file, save_feat_path)
