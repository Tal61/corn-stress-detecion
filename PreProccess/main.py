from spectral import *
import numpy as np
import datetime
import copy
import cv2
import statistics
import os.path as path
from tempfile import mkdtemp
from PIL import Image
# from numpy import zeros
from matplotlib import pyplot as plt
import os
import glob
import pandas as pd
import random
import shutil
import pre_proccess_functions as fn
import meta_data as md

storage_path = "D:/My Drive/StoragePath"
result_path = storage_path + "/ExpResults/tal_exp_results"
file_result_path = result_path + "/results"
data_set_path = storage_path + "/Datasets/tal_datesets/corn_data_set"
padding_w = 847
padding_h = 320
channel_750_index = md.Wavelength.index(750.08)
channel_680_index = md.Wavelength.index(680.06)
df_img_stats = pd.DataFrame(
    columns=['PlotName', 'NormalizedOver1', 'Over1Percent', 'NormalizedUnder0', 'Under0Percent', 'imgSize', 'MaxValue',
             'MinValue'])
df = pd.read_csv(f"{data_set_path}/phenotyping.csv")
df_max_ratio = pd.DataFrame(columns=range(731))
df["x_crop"] = ""
dates = ["200101"]  # ["191222","191224","191225","191229","191230","191231","200101"]
img_names = fn.get_dates_full_path(data_set_path, dates)
total_bands = 840
mask_directory = result_path + "/masks"
cropped_rgb_dir = result_path + "/cropped_RGB"
cropped_dir = result_path + "/cropped"
contours_dir = result_path + "/contours"
white_contour_dir = result_path + "/white_contours"
white_signature_dir = result_path + "/white_signature"
white_contour_dir1 = result_path + "/white_contours1"
white_hist_dir = result_path + "/white_hist"
pre_cropped = result_path + "/pre_cropped"
test_dir = result_path + "/test"
normalized_img_dir = result_path + "/mormalized"
normalized_RGBimg_dir = result_path + "/mormalizedRGB"
spectral_signature_dir = result_path + "/spectral_signature"
ndvi_img_dir = result_path + "/ndvi_img"
ndvi_hist_dir = result_path + "/ndvi_hist"
black_dir = result_path + "/black"
no_note_background_rgb_dir = result_path + "/no_note_background_rgb"
no_note_background_dir = result_path + "/no_note_background"
SNR = []
snr_threshold = 730
total_bands = snr_threshold
cropping_points = {
    191222: [0, 970, 115, 544],
    191224: [0, 970, 154, 544],
    191225: [0, 970, 115, 505],
    191229: [0, 970, 115, 505],
    191230: [0, 970, 60, 450],
    191231: [0, 970, 154, 544],
    200101: [0, 970, 154, 544],
}

# main
if __name__ == "__main__":
    plt.figure()
    for count, img_name in enumerate(img_names):
        if "Corn_plot8_2019-12-22_08-06-31" in img_name:
            continue
        # if count > 5:
        #     img_name = img_name
        img_directory = img_name.split('\\')[0]
        plot_name = img_name.split('\\')[1].replace(".hdr", "")  # "Corn_plot8_2019-12-22_08-06-31"
        print(plot_name)
        if ("black" in img_name) or ("bleck" in img_name):
            dark_image = envi.open(img_name)
            avg_dark_image, std_dark_image = fn.avg_dark_img(dark_image, black_dir, plot_name)
        elif "check" in img_name:
            continue
        else:
            date = int(dates[0])
            plot_num = int(plot_name.split('plot')[1].split("_")[0])
            img_hyper = envi.open(img_name) # envi.open('D:/My Drive/StoragePath/Datasets/tal_datesets/corn_data_set/191222/VNIR/191222/Corn_plot8_2019-12-22_08-06-31/capture\\Corn_plot8_2019-12-22_08-06-31.hdr')
            # SNR Threshold
            img_hyper = img_hyper[:, :, 0:snr_threshold]
            # if "_plot100_" in img_name:
            # cv2.imwrite('plot10_wave123.tif', img_hyper[:, :, 123])
            # cv2.imwrite('plot10_wave145.tif', img_hyper[:, :, 145])
            # save a temporary RGB
            #         save_rgb("RGB.png", img_hyper, [430, 179 + 20, 108])
            # crop
            img_hyper = fn.crop_image(img_hyper, cropping_points[date][0], cropping_points[date][1],
                                      cropping_points[date][2], cropping_points[date][3], pre_cropped, plot_name, True)
            plant_contour = fn.find_plant_contour(img_hyper, mask_directory, plot_name, contours_dir, True)
            cropped_img_hyper, x_crop, cropped_dark = fn.remove_background_2(img_hyper, plant_contour, img_directory,
                                                                             plot_name,
                                                                             cropped_rgb_dir, cropped_dir,
                                                                             avg_dark_image, channel_680_index,
                                                                             channel_750_index, True)
            df.loc[(df['SampleDate'] == date) & (df['plot'] == plot_num), "x_crop"] = x_crop
            # physical normalize
            if 'white_array' in locals():
                white_array, white_center = fn.white_circle_contour2(img_hyper, img_directory, plot_name,
                                                                     white_contour_dir,
                                                                     white_hist_dir, white_contour_dir1,
                                                                     white_contour_dir,
                                                                     white_array, True)
            else:
                white_array, white_center = fn.white_circle_contour2(img_hyper, img_directory, plot_name,
                                                                     white_contour_dir,
                                                                     white_hist_dir, white_contour_dir1,
                                                                     white_contour_dir,
                                                                     is_save_img=True)
            # SNR
            ratio_of_max = fn.white_max_ratio(white_array, plot_num)
            df_max_ratio = df_max_ratio.append(pd.DataFrame(ratio_of_max).T)
            df_max_ratio.iloc[-1, -1:] = plot_num
            curr_SNR = fn.get_snr(img_hyper, white_center, 10)  # shape 840X1
            SNR.append(curr_SNR)
            # normalize
            img_hyper, df_img_stats = fn.normalized_img2(white_array, cropped_img_hyper, cropped_dark, img_directory,
                                                         plot_name, normalized_RGBimg_dir, normalized_img_dir,
                                                         df_img_stats, channel_750_index, channel_680_index, True)

            no_note_image = fn.remove_note_background(img_hyper, channel_680_index, channel_750_index, plot_name,
                                                      ndvi_hist_dir, ndvi_img_dir, std_dark_image,
                                                      no_note_background_rgb_dir, no_note_background_dir,
                                                      cropped_img_hyper, padding_w, padding_h, is_save_img=True)
            # fn.avg_spectral_signature(img_hyper, spectral_signature_dir, plot_name)
    SNR = np.asarray(SNR)  # shape IageNUMX840
    SNR_mean = np.mean(SNR, axis=0)
    np.savetxt(file_result_path + "/" + "SNR" + plot_name + ".csv", SNR, delimiter=",")
    np.savetxt(file_result_path + "/" + "SNR_mean" + plot_name + ".csv", SNR_mean, delimiter=",")
    df_img_stats = df_img_stats.append(df_img_stats.agg(["mean"]))
    df_img_stats = df_img_stats.append(df_img_stats.agg(["std"]))
    df_img_stats = df_img_stats.append(df_img_stats.agg(["max"]))
    df_img_stats = df_img_stats.append(df_img_stats.agg(["min"]))
    file_name = 'PreProcessStats_' + dates[0] + '.csv'
    df_img_stats.to_csv(file_result_path + "/" + file_name, index=False)
    df_max_ratio["total_over_0.5"] = np.sum(df_max_ratio.iloc[:, :-1] > 0.5, axis=0)
    file_name = 'max_white_ratio_' + dates[0] + '.csv'
    df_max_ratio.to_csv(file_result_path + "/" + file_name, index=False)

