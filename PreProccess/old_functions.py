def find_white_contour(hyperspectral_image, img_directory, plot_name, is_save_img=True):
    # img_directory = img_name.split('\\')[0]
    # plot_name = img_name.split('\\')[1]

    normalized_img = np.zeros(
        (hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]))
    ## lower_green = np.array([0, 0, 212])
    lower_white = np.array([50, 50, 50])
    ## upper_green = np.array([131, 255, 255])
    upper_white = np.array([9000, 9000, 9000])
    save_rgb("RGB.png", hyperspectral_image, [430, 179 + 20, 108])
    frame = cv2.imread("RGB.png")

    ### HSV green filter - binary image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)  # set pixels out of range -0, in range - 255

    ### Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    contour = contours[max_index]
    cv2.drawContours(frame, contour, -1, (238, 11, 11), 2)
    # cv2.imwrite(img_directory + "/"  + "_white_contour.png", frame)
    # cv2.imwrite(img_directory + "/"  + plot_name + "_white_contour.png", frame)
    cv2.imwrite(white_contour_dir1 + "/" + plot_name + "_white_contour.png", frame)

    #### create white image
    vector = []
    (x, y, w, h) = cv2.boundingRect(contour)  # get large boundings
    for i in range(y, (y + round(h / 2) + 1)):
        for j in range(x, (x + round(w / 2) + 1)):
            if cv2.pointPolygonTest(contour, (j, i),
                                    True) >= 0:  ##  Negative value if the point is outside the contour. gives actual boundings
                vector.append(hyperspectral_image[i, j, :])  # all bands of white pixels
                normalized_img[i, j, :] = hyperspectral_image[i, j, :]  # create only white image
    # save_rgb(img_directory + "/"  + plot_name + "_examine_white_contour_RGB.png", normalized_img, [430, 179 + 20, 108])
    save_rgb(white_contour_dir + "/" + plot_name + "_examine_white_contour_RGB.png", normalized_img,
             [430, 179 + 20, 108])
    array = np.asarray(vector, )
    array = array.reshape(-1, array.shape[3])
    band_white_histogram(array, 100, img_directory, plot_name)  # save hist of whitepixels in band 100
    #     band_white_histogram(array, 200)
    #     band_white_histogram(array, 300)
    #     band_white_histogram(array, 400)
    #     band_white_histogram(array, 500)
    #     band_white_histogram(array, 600)
    #     band_white_histogram(array, 700)
    #     band_white_histogram(array, 800)
    return (array)


def remove_background(hyperspectral_image, contour, image_dir, plot_name, path, path2, is_save_img=True):
    return_img = np.zeros((hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]))
    (x, y, w, h) = cv2.boundingRect(contour)
    print((x, y, w, h))
    for i in range(y, (y + h + 1)):
        for j in range(x, (x + w + 1)):
            if cv2.pointPolygonTest(contour, (j, i), True) >= 0:  ##  Negative value if the point is outside the contour
                return_img[i, j, :] = hyperspectral_image[i, j, :]
    return_img = return_img[y:(y + h), x:(x + w), :]
    if is_save_img:
        # save_rgb(image_dir + "/" + plot_name + "_remove_background2_RGB.png", return_img, [430, 179 + 20, 108])
        save_rgb(path + "/" + plot_name + "_remove_background_RGB.png", return_img, [430, 179 + 20, 108])
        # envi.save_image(image_dir + "/" + plot_name  + "_cropped2.hdr", return_img, dtype=np.float32)
        envi.save_image(path2 + "/" + plot_name + "_cropped.hdr", return_img, dtype=np.float32)
    return return_img


def white_circle_contour(hyperspectral_image, img_directory, plot_name, is_save_img=True):
    normalized_img = np.zeros(
        (hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]))
    save_rgb("RGB.png", hyperspectral_image, [430, 179 + 20, 108])
    # detect circles in the image
    X = cv2.imread("RGB.png")
    output = X.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    output = cv2.resize(output, (700, 544))
    circles = cv2.HoughCircles(output,
                               cv2.HOUGH_GRADIENT,
                               minDist=10,
                               dp=1.2
                               ,
                               param1=150,
                               param2=30,
                               minRadius=20,
                               maxRadius=40)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        #       for (x, y, r) in circles:
        # corresponding to the center of the circle
        (x, y, r) = circles[0]
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the circle
        #         plt.imshow(output)
        #         plt.show()
        # rescale
        x = int(x * original_x / 700)
        # draw elipse
        elipse = cv2.ellipse(X, (x, y), (int(r * (original_x / 700)), r), 0, 0, 360, (0, 255, 0), 1)
        # draw center
        white_contour = cv2.rectangle(elipse, (x - 1, y - 1), (x + 1, y + 1), (255, 0, 0), -1)
        cv2.imwrite(white_contour_dir1 + "/" + plot_name + "_white.png", white_contour)
        #         plt.imshow(white_contour)
    #         plt.show()

    ####get the pixels in the contour
    raw_img = cv2.imread('RGB.png')
    # create darek image
    dark = np.zeros((raw_img.shape[0], raw_img.shape[1], raw_img.shape[2]))
    # draw elipse on dark
    dark_elipse = cv2.ellipse(dark, (x, y), (int(r * (original_x / 700)), r), 0, 0, 360, (0, 255, 0), 1)
    # get elipse contours
    rows, cols, channel = np.where(dark_elipse > 0)
    contours = np.array((cols, rows)).T
    contours = contours.reshape([contours.shape[0], 1, contours.shape[1]])
    cv2.drawContours(raw_img, contours, -1, (238, 11, 11), 2)
    cv2.imwrite(white_contour_dir1 + "/" + plot_name + "_white1.png", white_contour)
    plt.imshow(raw_img)
    plt.show()

    #### create white image
    vector = []
    (x, y, w, h) = cv2.boundingRect(contours)  # get large boundings
    for i in range(y, y + h):
        for j in range(x, x + w):
            if cv2.pointPolygonTest(contours, (j, i),
                                    True) >= 0:  ##  Negative value if the point is outside the contour. gives actual boundings
                vector.append(hyperspectral_image[i, j, :])  # all bands of white pixels
                normalized_img[i, j, :] = hyperspectral_image[i, j, :]  # create only white image
    save_rgb(img_directory + "/" + plot_name + "_examine_white_contour_RGB.png", normalized_img, [430, 179 + 20, 108])
    save_rgb(white_contour_dir + "/" + plot_name + "_examine_white_contour_RGB.png", normalized_img,
             [430, 179 + 20, 108])
    array = np.asarray(vector, )
    array = array.reshape(-1, array.shape[3])
    band_white_histogram(array, 100, img_directory, plot_name)  # save hist of whitepixels in band 100
    return array

def normalized_img(hyperspectral_image, dark_image, img_directory, plot_name, is_save_img=True):
    white_array = find_white_contour(hyperspectral_image, img_directory, plot_name)
    avg_white_spectral_signature(white_array, img_directory, plot_name)

    H, W, B = hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]
    normalized_img = np.zeros((H, W, B))
    for b in range(total_bands):
        pic = hyperspectral_image[:, :, b]  # an image in one band dimension
        pic = pic.reshape(pic.shape[0], pic.shape[1])
        white_vector = white_array[:, b]  # the white part of the image in the same band
        # white = np.mean(white_vector)
        hist, bin_edges = np.histogram(white_vector, bins=100)  #
        max_index = np.argmax(hist[0:(len(hist) - 3)])
        white = bin_edges[max_index]  # get values of most common white
        dark = np.tile(avg_dark_image[0, :, b], (H, 1))  # expand average dark to pic Hight
        normalized_img[..., b] = np.divide(np.subtract(pic, dark), np.subtract((np.ones([H, W]) * white),
                                                                               dark))  # (Iraw - Idark)/(Iwhite-Idark)
        # cv2.imwrite(image_dir + "/" + plot_name  + "_normalized_band_100.jpg", normalized_img[...,100])
    return white_array  # normalized_img

def normalized_img_black_from_image(hyperspectral_image, img_directory, plot_name, is_save_img=True):
    white_array = find_white_contour(hyperspectral_image, img_directory, plot_name)
    avg_white_spectral_signature(white_array, img_directory, plot_name)
    black_values = get_black_values(hyperspectral_image)
    H, W, B = hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]
    normalized_img = np.zeros((H, W, B))
    for b in range(total_bands):
        pic = hyperspectral_image[:, :, b]  # an image in one band dimension
        pic = pic.reshape(pic.shape[0], pic.shape[1])
        white_vector = white_array[:, b]  # the white part of the image in the same band
        # white = np.mean(white_vector)
        hist, bin_edges = np.histogram(white_vector, bins=100)  #
        max_index = np.argmax(hist[0:(len(hist) - 3)])
        white = bin_edges[max_index]  # get values of most common white
        dark = np.tile(black_values[b], (H, W))  # expand average dark to pic Hight
        normalized_img[..., b] = np.divide(np.subtract(pic, dark), np.subtract((np.ones([H, W]) * white),
                                                                               dark))  # (Iraw - Idark)/(Iwhite-Idark)
        # cv2.imwrite(image_dir + "/" + plot_name  + "_normalized_band_100.jpg", normalized_img[...,100])
    return normalized_img

# def normalized_img2(hyperspectral_image, dark_image,img_directory,plot_name):
#     white_array = white_circle_contour2(hyperspectral_image,img_directory,plot_name)
#     avg_white_spectral_signature(white_array,img_directory,plot_name)
#     H, W, B = hyperspectral_image.shape[0], hyperspectral_image.shape[1], hyperspectral_image.shape[2]
#     normalized_img = np.zeros((H, W, B))
#     for b in range(B):
#         pic = hyperspectral_image[:,:, b]   #an image in one band dimension
#         pic = pic.reshape(pic.shape[0],pic.shape[1])
#         white_vector = white_array[:, b]   # the white part of the image in the same band
#         # white = np.mean(white_vector)
#         hist, bin_edges = np.histogram(white_vector, bins=100)  #
#         max_index = np.argmax(hist[0:(len(hist)-3)])
#         white = bin_edges[max_index]                 #get values of most common white
#         dark = np.tile(avg_dark_image[0, :, b], (H, 1))  #expand average dark to pic Hight
#         normalized_img[..., b] = np.divide(np.subtract(pic, dark), np.subtract((np.ones([H, W]) * white), dark)) #(Iraw - Idark)/(Iwhite-Idark)
#          #cv2.imwrite(image_dir + "/" + plot_name  + "_normalized_band_100.jpg", normalized_img[...,100])
#     return white_array #normalized_img