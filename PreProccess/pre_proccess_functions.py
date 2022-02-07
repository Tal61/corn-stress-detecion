
import numpy as np
import datetime
import copy
import cv2
import statistics
import os.path as path
import os
import glob
import pandas as pd
import random
import shutil
import ctypes  # An included library with Python install.

from spectral import *
from tempfile import mkdtemp
from PIL import Image
# from numpy import zeros
from matplotlib import pyplot as plt


storage_path = "D:/My Drive/StoragePath"
result_path = storage_path + "/ExpResults/tal_exp_results"
file_result_path = result_path + "/results"
data_set_path = storage_path + "/Datasets/tal_datesets/corn_data_set"


df_img_stats = pd.DataFrame(
    columns=['PlotName', 'NormalizedOver1', 'Over1Percent', 'NormalizedUnder0', 'Under0Percent', 'imgSize', 'MaxValue',
             'MinValue'])
df = pd.read_csv(f"{data_set_path}/phenotyping.csv")
df["x_crop"] = ""
# dates = ["191222"]  # ["191222","191224","191225","191229","191230","191231","200101"]
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
black_hist_dir = result_path + "/dark_hist"
normalized_hist_dir = result_path + "/normalized_pixel_hist"

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


def load_images(dates, rgb=False):
    # load images
    path_hyper = "D:/Hyperspectral"
    # dates = ["191224"]
    img_names = []
    Y = pd.read_csv(path_hyper + "\phenotyping.csv")
    X = []
    for d in dates:
        for dirs in os.listdir(path_hyper + "/" + d + "/VNIR/" + d):
            if not rgb:
                fullpath = path_hyper + "/" + d + "/VNIR/" + d + "/" + dirs + "/capture"
                fullpath = glob.glob(fullpath + "/*.hdr")
                img_names.append(fullpath[0])
            else:
                fullpath = path_hyper + "/" + d + "/VNIR/" + d + "/" + dirs
                fullpath = glob.glob(fullpath + "/*.png")
                img_names.append(fullpath[0])

    for i, img in enumerate(img_names):
        if rgb == False:
            img_hyper = open_image(img)  # envi.open(img)   #.load()
        else:
            img_hyper = Image.open(img)
        X.append(img_hyper)
        # save_rgb(img_paths[i] + "/RGB.png", img_hyper, [430, 179 + 20, 108])   # save RGB
    return X


def create_folder_images_test_train(dates, image_type):
    # load images
    Path = "D:/Hyperspectral"
    img_names = []
    Y = pd.read_csv(Path + "\phenotyping.csv")
    for d in dates:
        for dirs in os.listdir(Path + "/" + d + "/VNIR/" + d):
            if "black" in dirs:
                continue
            if image_type == "hdr":
                fullpath = Path + "/" + d + "/VNIR/" + d + "/" + dirs + "/capture"
                fullpath = glob.glob(fullpath + "/*.hdr")
                img_names.append(fullpath[0])
            elif image_type == "rgb":
                fullpath = Path + "/" + d + "/VNIR/" + d + "/" + dirs
                fullpath = glob.glob(fullpath + "/*.png")
                img_names.append(fullpath[0])
            elif image_type == "cropped_rgb":
                fullpath = Path + "/" + d + "/VNIR/" + d + "/" + dirs + "/capture"
                fullpath = glob.glob(fullpath + "/*remove_background_RGB.png")
                img_names.append(fullpath[0])
    for i, img in enumerate(img_names):
        original = img
        target = 'D:/Hyperspectral/RGB_images'
        if random.random() > 0.3:
            target = target + "/Train"
        else:
            target = target + "/Test"
        date = int(img.split('/')[2])
        img_name = img.split('\\')[1]
        if ("plot" in img_name) == False:
            continue
        plot_num = int(img_name.split('plot')[1].split("_")[0])
        if plot_num > 0:
            # print(date)
            print(plot_num)
            label = Y[(Y['SampleDate'] == date) & (Y['plot'] == plot_num)]['Y'].iloc[0]
            if label == 0:  # Healthy
                target = target + "/Healthy"
            elif label == 1:  # Sick
                target = target + "/Sick"
            else:
                continue
        else:
            continue
        target += "/" + img.split('\\')[1]
        shutil.copyfile(original, target)


def get_dates_full_path(path, dates, rgb=False):
    # load images
    img_names = []
    X = []
    for d in dates:
        for dirs in os.listdir(path + "/" + d + "/VNIR/" + d):
            if rgb == False:
                fullpath = path + "/" + d + "/VNIR/" + d + "/" + dirs + "/capture"
                fullpath = glob.glob(fullpath + "/*.hdr")
                img_names.append(fullpath[0])
            else:
                fullpath = path + "/" + d + "/VNIR/" + d + "/" + dirs
                fullpath = glob.glob(fullpath + "/*.png")
                img_names.append(fullpath[0])
    return img_names


def find_plant_contour(img_hyper, img_directory, plot_name, path, is_save_img=True):
    # img_directory = img_name.split('\\')[0]
    save_rgb(result_path + "/RGB.png", img_hyper, [430, 179 + 20, 108])
    frame = cv2.imread(result_path + "/RGB.png")
    ### HSV green filter - binary image
    lower_green = np.array([60 - 50, 50, 15])
    upper_green = np.array([60 + 10, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert to HSV
    # cv2.imwrite(plot_name + "_HSV.png", frame)  # save HSV
    mask = cv2.inRange(hsv, lower_green, upper_green)  # set pixels out of range -0, in range - 255
    # cv2.imwrite(img_directory + "/"  + plot_name + "_mask2.png", mask)   #save mask
    cv2.imwrite(img_directory + "/" + plot_name + "_mask.png", mask)  # save mask
    ### Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # find the contours
    areas = [cv2.contourArea(c) for c in contours]  # cal the area of the contours
    max_index = np.argmax(areas)  # the contour that contains max area is the plant
    contour = contours[max_index]
    cv2.drawContours(frame, contour, -1, (255, 255, 255), 1)
    if is_save_img:
        # cv2.imwrite(img_directory + "/" + plot_name + "_contour2.png", frame)   #save mask
        cv2.imwrite(path + "/" + plot_name + "_contour.png", frame)  # save mask
    return contour


def band_white_histogram(array, b, image_dir, plot_name, is_save_img=True):
    # array = array.reshape(-1,array.shape[3])
    vector_b = array[:, b]
    hist, bin_edges = np.histogram(vector_b, bins=100)
    max_index = np.argmax(hist[0:(len(hist) - 3)])
    white = bin_edges[max_index]
    # plt.figure()
    plt.cla()
    plt.hist(vector_b, bins=100)  # calculating histogram
    plt.xlabel('Pixel values')
    plt.ylabel('No. of pixels')
    plt.title('white histogram plot: ' + plot_name + ' band: ' + b.__str__() + ' peak value: ' + white.__str__())
    if is_save_img:
        plt.savefig(image_dir + "/" + plot_name + "_white_histogram_plot" + '_band_' + b.__str__() + '.jpg')


def white_circle_contour2(hyperspectral_image, img_directory, plot_name, path, path_hist, white_contour_dir1,
                          white_contour_dir, white_array=0, is_save_img=True):
    if "Corn_plot8_2019-12-22" in plot_name:
        h_start = 110
        h_end = 130
        w_start = 660
        w_end = 687
        center = (w_start + w_end) // 2, (h_start + h_end) // 2
        # plt.imshow(hyperspectral_image[h_start:h_end, w_start:w_end, 100])
        # plt.show()
        white_array = hyperspectral_image[h_start:h_end, w_start:w_end]
        bands = white_array.shape[2]
        white_array = white_array.transpose(2, 0, 1).reshape(bands, -1).transpose(1, 0)
        return white_array, center
    normalized_img = np.zeros(
        (hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]))
    normalized_img2 = np.zeros(
        (hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]))
    save_rgb(result_path + "/RGB.png", hyperspectral_image, [430, 179 + 20, 108])
    # detect circles in the image
    X = cv2.imread(result_path + "/RGB.png")
    output1 = X.copy()
    original_y = X.shape[0]
    original_x = X.shape[1]
    resized_x = 400
    output2 = cv2.cvtColor(output1, cv2.COLOR_BGR2GRAY)

    output = cv2.resize(output2, (resized_x, original_y))
    circles = cv2.HoughCircles(output,
                               cv2.HOUGH_GRADIENT,
                               minDist=10,
                               dp=1.2
                               ,
                               param1=150,
                               param2=30,
                               minRadius=25,
                               maxRadius=45)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        (x, y, r) = circles[0]
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # cv2.imshow("white resize", output)
        # rescale
        x = int(x * original_x / resized_x)
        # draw elipse
        elipse = cv2.ellipse(X, (x, y), (int(r * (original_x / resized_x)), r), 0, 0, 360, (0, 255, 0), 1)
        # draw center
        white_contour = cv2.rectangle(elipse, (x - 1, y - 1), (x + 1, y + 1), (255, 0, 0), -1)
        # cv2.imshow("white resize", white_contour)
        if is_save_img:
            cv2.imwrite(white_contour_dir1 + "/" + plot_name + "_white.png", white_contour)
        white_center = (x, y)
        ####get the pixels in the contour
        raw_img = cv2.imread(result_path + "/RGB.png")
        # create darek image
        dark = np.zeros((raw_img.shape[0], raw_img.shape[1], raw_img.shape[2]))
        # draw elipse on dark
        dark_elipse = cv2.ellipse(dark, (x, y), (int(r * (original_x / resized_x)), r), 0, 0, 360, (0, 255, 0), 1)
        # get elipse contours
        rows, cols, channel = np.where(dark_elipse > 0)
        contours = np.array((cols, rows)).T
        contours = contours.reshape([contours.shape[0], 1, contours.shape[1]])
        cv2.drawContours(raw_img, contours, -1, (238, 11, 11), 2)
        if is_save_img:
            cv2.imwrite(white_contour_dir1 + "/" + plot_name + "_white.png", white_contour)
        # plt.imshow(raw_img)
        # plt.show()

        #### create white image
        #    vector = []
        (x, y, w, h) = cv2.boundingRect(contours)  # get large boundings
        for i in range(y, y + h):
            for j in range(x, x + w):
                if cv2.pointPolygonTest(contours, (j, i),
                                        True) >= 0:  ##  Negative value if the point is outside the contour. gives actual boundings
                    #               vector.append(hyperspectral_image[i, j, :])    # all bands of white pixels
                    normalized_img[i, j, :] = hyperspectral_image[i, j, :]  # create only white image
        if is_save_img:
            #     save_rgb(img_directory + "/"  + plot_name + "_examine_white_contour_RGB.png", normalized_img, [430, 179 + 20, 108])
            save_rgb(white_contour_dir + "/" + plot_name + "_examine_white_contour_RGB.png", normalized_img,
                     [430, 179 + 20, 108])
        save_rgb(result_path + "/white.png", normalized_img, [430, 179 + 20, 108])
        image = cv2.imread(result_path + "/white.png", cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #   save_rgb("white1.png", normalized_img, [430, 179 + 20, 108])
        #   plt.imshow(binary)
        indices = np.argwhere(binary > 0)
        vector = []

        for pixel in indices:
            vector.append(normalized_img[pixel[0], pixel[1], :])
            normalized_img2[pixel[0], pixel[1], :] = normalized_img[pixel[0], pixel[1], :]
        if is_save_img:
            save_rgb(path + "/" + plot_name + "_examine_white_contour_RGB2.png", normalized_img2, [430, 179 + 20, 108])
        array = np.asarray(vector, )

        # array = array.reshape(-1,array.shape[3])
        band_white_histogram(array, 121, path_hist, plot_name)  # save hist of whitepixels in band 100
        band_white_histogram(array, 124, path_hist, plot_name)
        band_white_histogram(array, 128, path_hist, plot_name)
        band_white_histogram(array, 144, path_hist, plot_name)
        band_white_histogram(array, 148, path_hist, plot_name)
        band_white_histogram(array, 137, path_hist, plot_name)
    else:
        # ctypes.windll.user32.MessageBoxW(0, "no circle found in image " + plot_name, "Your title", 1)
        print("no circle found in image " + plot_name)
        array = white_array
    return array, white_center


def white_max_ratio(white_array, plot_name):
    max_all_bands = np.max(white_array, axis=0)  #
    total_max_pixels = np.sum(white_array == max_all_bands.reshape(1, -1), axis=0)
    ratio = total_max_pixels / np.count_nonzero(white_array, axis=0)
    return ratio


def avg_white_spectral_signature(white_array, image_dir, plot_name, is_save_img=True):
    list = []
    # white_array = white_array.reshape(-1,white_array.shape[3])
    for b in range(total_bands):
        white_vector = white_array[:, b]
        hist, bin_edges = np.histogram(white_vector, bins=100)
        max_index = np.argmax(hist[0:(len(hist) - 3)])
        list.append(bin_edges[max_index])
    # avg = np.mean(white_array, axis=0)
    # plt.figure()
    plt.plot(list)
    plt.xlabel('band')
    plt.ylabel('reflectance')
    plt.title("common value white spectral signature")
    plt.ylim(0, 4500)
    if is_save_img:
        plt.savefig(image_dir + "/" + plot_name + '_common_white_spectral_signature.jpg')
        plt.savefig(white_signature_dir + "/" + plot_name + '_common_white_spectral_signature.jpg')


# returns average black per band.  shape (1,1600, total_bands)
def avg_dark_img(dark_image, img_directory, plot_name, is_save_img=True):
    avg_dark_img = np.zeros((1, dark_image.shape[1], dark_image.shape[2]))
    std_dark_img = np.zeros((1, dark_image.shape[1], dark_image.shape[2]))
    # std_per_band_dark_img = np.zeros((1, dark_image.shape[2]))
    H, W, B = dark_image.shape[0], dark_image.shape[1], dark_image.shape[2]
    for b in range(B):
        # print("band: ", b)
        temp = dark_image[:, :, b]  ## band image
        temp_mean = (np.mean(temp, axis=0)).transpose()  ## vector of means
        avg_dark_img[:, :, b] = temp_mean
        std_dark_img[:, :, b] = (np.std(temp, axis=0)).transpose()
        # std_per_band_dark_img[:, :, b] = (np.std(temp, axis=(0,1))
        # dark_img_new[:, :, b] = np.tile(temp_mean, (H,1))
    avg_dark_img_for_rgb = avg_dark_img.reshape(avg_dark_img.shape[1], avg_dark_img.shape[2])
    if is_save_img:
        save_rgb(img_directory + "/" + plot_name + "_average_dark.png", avg_dark_img_for_rgb)
    return avg_dark_img, std_dark_img


# def get_black_values(img_hyper):
#     flat_image = np.reshape(img_hyper[:, :, :], (-1, total_bands))
#     black_value = np.zeros(total_bands)
#     for b in range(total_bands):
#         cur_flat_image = flat_image[:, b]
#         precetaile10 = np.percentile(cur_flat_image, 10, axis=0)
#         precetaile20 = np.percentile(cur_flat_image, 20, axis=0)
#         mask = (cur_flat_image > precetaile10) & (cur_flat_image < precetaile20)
#         black_array = cur_flat_image[mask]
#         cur_black_value = np.mean(black_array)
#         black_value[b] = cur_black_value
#     return black_value


def normalized_img2(white_array, hyperspectral_image, dark_image, img_directory, plot_name, path_rgb, path,
                    df_img_stats, channel_750_index, channel_680_index, is_save_img=True):
    # white_array = white_circle_contour2(hyperspectral_image,img_directory,plot_name)
    avg_white_spectral_signature(white_array, img_directory, plot_name)
    H, W, B = hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]
    normalized_img = np.zeros((H, W, B))
    for b in range(B):
        pic = hyperspectral_image[:, :, b]  # an image in one band dimension
        pic = pic.reshape(pic.shape[0], pic.shape[1])
        white_vector = white_array[:, b]  # the white part of the image in the same band
        white = np.median(white_vector)
        dark = np.tile(dark_image[0, :, b], (H, 1))  # expand average dark to pic Hight
        normalized_img[..., b] = np.divide(np.subtract(pic, dark), np.subtract((np.ones([H, W]) * white),
                                                                               dark))  # (Iraw - Idark)/(Iwhite-Idark)
    if is_save_img:
        save_rgb(path_rgb + "/" + plot_name + "NormalizedRGB.png", normalized_img, [430, 179 + 20, 108])
        envi.save_image(path + "/" + plot_name + "normalized.hdr", normalized_img, dtype=np.float32)
        # cv2.imwrite(normalized_img_dir + "/" + plot_name  + "_normalized_band_100.jpg", normalized_img)
    # statistical check
    total_pixels_over_1 = np.sum(normalized_img > 1)
    over_1_precentage = total_pixels_over_1 / normalized_img.size
    total_pixels_under_0 = np.sum(normalized_img < 0)
    under_0_precentage = total_pixels_under_0 / normalized_img.size
    black_hist_in_band(dark_image, channel_750_index, plot_name)
    black_LineChart_in_band(dark_image, channel_750_index, plot_name)
    normalized_hist(normalized_img, plot_name)
    normalized_hist(normalized_img[:, :, channel_750_index], plot_name + 'channel_750')
    normalized_hist(normalized_img[:, :, channel_680_index], plot_name + 'channel_680')
    df_img_stats = df_img_stats.append(
        {'PlotName': plot_name, 'NormalizedOver1': total_pixels_over_1, 'Over1Percent': over_1_precentage,
         'NormalizedUnder0': total_pixels_under_0, 'Under0Percent': under_0_precentage,
         'imgSize': (normalized_img.size * 4) / 1000000, 'MaxValue': normalized_img.max(),
         'MinValue': normalized_img.min()},
        ignore_index=True)
    normalized_img[normalized_img > 1] = 1
    normalized_img[normalized_img < 0] = 0
    return normalized_img, df_img_stats


def black_hist_in_band(dark_image, band, plot_name):
    # plt.figure()
    plt.cla()
    plt.hist(dark_image[:, :, band].reshape([-1]), 1000)  # calculating histogram
    plt.xlabel('Dark Pixel values')
    plt.ylabel('No. of pixels')
    plt.title('dark histogram plot: ' + plot_name)
    plt.savefig(black_hist_dir + "/" + plot_name + "dark_hist_band_" + str(band) + '.jpg')


def black_LineChart_in_band(dark_image, band, plot_name):
    plt.cla()
    y_val = dark_image[:, :, band].reshape([-1])
    x = np.arange(len(y_val))
    # plt.figure()
    plt.bar(x, y_val, align='center', alpha=0.5)
    plt.xlabel('Width pixels')
    plt.ylabel('average value')
    plt.title('dark histogram plot: ' + plot_name)
    plt.savefig(black_hist_dir + "/" + plot_name + "dark_bar_band_" + str(band) + '.jpg')


def normalized_hist(normalized_img, plot_name):
    # plt.figure()
    plt.cla()
    plt.hist(normalized_img.reshape([-1]), 1000, [-0.5, 1])  # calculating histogram
    plt.xlabel('Pixel values')
    plt.ylabel('No. of pixels')
    plt.title('normalized histogram plot: ' + plot_name)
    # if is_save_img:
    plt.savefig(normalized_hist_dir + "/" + plot_name + "_normalized_pixel_hist" + '.jpg')


def create_hist(img, plot_name, title, dir, bins, is_save_img=True):
    # plt.figure()
    plt.cla()
    plt.hist(img.reshape([-1]), bins=bins)  #
    plt.xlabel('Pixel values')
    plt.ylabel('No. of pixels')
    plt.title(title + ': ' + plot_name)
    if is_save_img:
        plt.savefig(dir + "/" + plot_name + title + '.jpg')


def avg_spectral_signature(hyperspectral_image, image_dir, plot_name, is_save_img=True):
    avg = np.sum(np.sum(hyperspectral_image[:, :, :], axis=0), axis=0) / np.count_nonzero(hyperspectral_image[:, :, 1])
    # plt.figure()
    plt.plot(avg)
    plt.xlabel('band')
    plt.ylabel('normalized reflectance')
    plt.title("average plant spectral signature")
    plt.ylim(0, 1)
    if is_save_img:
        plt.savefig(image_dir + "/" + plot_name + '_avg_spectral_signature.jpg')


def crop_image(hyperspectral_image, start_x, end_x, start_y, end_y, save_dir, plot_name, is_save_img=True):
    save_rgb(result_path + "/RGB.png", hyperspectral_image, [430, 179 + 20, 108])
    image = cv2.imread(result_path + "/RGB.png")  # hyperspectral_image[:,:,[430, 179 + 20, 108]]#
    imCrop = image[start_y:end_y, start_x:end_x]
    imCropHyperSpectral = hyperspectral_image[start_y:end_y, start_x:end_x]
    # cv2.imwrite(save_dir + "/" + plot_name + "_white_contour.png", imCrop)
    #     save_rgb(result_path + "/RGB.png", img_hyper, [430, 179 + 20, 108])
    if is_save_img:
        save_rgb(save_dir + "/" + plot_name + "_pre_cropped.png", imCropHyperSpectral, [430, 179 + 20, 108])
    # plt.imshow(imCrop)
    return imCropHyperSpectral


def remove_background_2(hyperspectral_image, contour, image_dir, plot_name, path, path2, dark, channel_680_index,
                        channel_750_index, is_save_img=True):
    bigger_by = 0.1
    return_img = np.zeros((hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]))
    (x, y, w, h) = cv2.boundingRect(contour)
    #    print((x, y, w, h))
    #     for i in range(y, (y+h+1)):
    #         for j in range(x, (x+w+1)):
    #             if cv2.pointPolygonTest(contour, (j, i), True) >= 0:  ##  Negative value if the point is outside the contour
    #                 return_img[i, j, :] = hyperspectral_image[i, j, :]
    y_start = y - round(bigger_by * h)
    y_end = y + round((1 + bigger_by) * h)
    x_start = x - round(bigger_by * w)
    x_end = x + round((1 + bigger_by) * w)
    #    print((x_start, x_end, y_start, y_end))
    if y_start < 0:
        y_start = 0
    if y_end > hyperspectral_image.shape[0]:
        y_end = hyperspectral_image.shape[0]
    if x_start < 0:
        x_start = 0
    if x_end > hyperspectral_image.shape[1]:
        x_end = hyperspectral_image.shape[1]
    # return_img = hyperspectral_image[y:(y+round(1.1*h)), x:(x+round(1.1*w)), :]
    return_img = hyperspectral_image[y_start:y_end, x_start:x_end, :]
    if "_plot10_" in plot_name:
        cv2.imwrite('plot10Cropped_wave750.tif', return_img[:, :, channel_750_index])
        cv2.imwrite('plot10Cropped_wave680.tif', return_img[:, :, channel_680_index])
    cropped_dark = dark[:, x_start:x_end, :]
    if is_save_img:
        # save_rgb(image_dir + "/" + plot_name + "_remove_background2_RGB.png", return_img, [430, 179 + 20, 108])
        save_rgb(path + "/" + plot_name + "_remove_background_RGB.png", return_img, [430, 179 + 20, 108])
        # envi.save_image(image_dir + "/" + plot_name  + "_cropped2.hdr", return_img, dtype=np.float32)
        envi.save_image(path2 + "/" + plot_name + "_cropped.hdr", return_img, dtype=np.float32)
    x_crop = (x_start + x_end) / 2
    return return_img, x_crop, cropped_dark


def get_snr(hyperspectral_image, white_center, size):
    x, y = white_center
    half_size = int(size / 2)
    x_start = int(x - half_size)
    x_end = int(x + half_size)
    y_start = int(y - half_size)
    y_end = int(y + half_size)
    white_10x10 = hyperspectral_image[y_start:y_end, x_start:x_end, :]
    mean = np.mean(white_10x10, axis=(0, 1))
    std = np.std(white_10x10, axis=(0, 1))
    snr = mean / std
    return snr


def remove_note_background(normalized_img, channel_680_index, channel_750_index, plot_name, ndvi_hist_dir, ndvi_img_dir,
                           std_dark_image, no_note_background_rgb_dir, no_note_background_dir, cropped_img_hyper, padding_w, padding_h,
                           is_save_img=True):
    ndvi_img = ndvi(normalized_img, channel_680_index, channel_750_index)
    if is_save_img:
        cv2.imwrite(ndvi_img_dir + "/" + plot_name + "_NDVI.tif", ndvi_img)
    if is_save_img:
        title = "_NDVI_hist_"
        create_hist(ndvi_img, plot_name, title, ndvi_hist_dir, 50, is_save_img)
    # calculate background to inject
    mean = np.median(normalized_img[:, 0:1, :], axis=(0, 1))  # median for every band
    black_std = np.std(normalized_img[:, 0:1, :], axis=(0, 1))
    indices = np.argwhere(ndvi_img < 0.4)
    # add for padding - ht = normalized_img.shape[0]
    # add for padding - wd = normalized_img.shape[1]
    # add for padding - final_image = np.zeros((padding_h, padding_w, normalized_img.shape[2]))
    for band in range(normalized_img.shape[2]):
        random_array = np.random.normal(mean[band], black_std[band], indices.shape[0])
        for count, pixel in enumerate(indices):
            normalized_img[pixel[0], pixel[1], band] = random_array[count]
            # cropped_img_hyper[pixel[0], pixel[1], band] = random_array[count]
        # """add padding"""
        # padded_img = np.random.normal(mean[band], black_std[band], (padding_h, padding_w))
        # # compute center offset
        # xx = (padding_w - wd) // 2
        # yy = (padding_h - ht) // 2
        # # enter image
        # padded_img[yy:yy + ht, xx:xx + wd] = normalized_img[:, :, band]
        # final_image[:, :, band] = padded_img
    if is_save_img:
        envi.save_image(no_note_background_dir + "/" + plot_name + "_no_note.hdr", normalized_img, dtype=np.float32)
        save_rgb(no_note_background_rgb_dir + "/" + plot_name + "_no_note_RGB4.png", normalized_img,
                 [430, 179 + 20, 108])
    if is_save_img:
        title = "_no_note_hist_"
        create_hist(normalized_img[:, :, 498], plot_name, title + "_channel_750", ndvi_hist_dir, 90, is_save_img)
        create_hist(normalized_img[:, :, 498], plot_name, title + "with_background_channel_750", ndvi_hist_dir, 90, is_save_img)
    return normalized_img
