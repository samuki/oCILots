# code from https://github.com/ardaduz/cil-road-segmentation/tree/master/additional-data

import sys
import cv2
import io
import glob
import numpy as np
import pandas as pd
from scipy import stats
from urllib import request
from PIL import Image


def input_check():
    filtered_data = dict()
    all_data = pd.read_csv("input_cities.csv", index_col='city').to_dict(orient='index')
    for city in all_data:
        values = all_data[city]
        filtered_data[city] = values
    return True, filtered_data


def get_random_locations(city, n_locations=None):
    """
    :param city: a dict element containing the bounding box information and number of images
    :return: N x 2 numpy array with lat-lon pairs (N = n-images)
    """
    if n_locations is None:
        n_locations = city['n-images']
    lat_min = city['latitude-min']
    lat_max = city['latitude-max']
    lon_min = city['longitude-min']
    lon_max = city['longitude-max']

    lat_samples = np.random.uniform(lat_min, lat_max, n_locations)
    lon_samples = np.random.uniform(lon_min, lon_max, n_locations)
    location_samples = np.transpose(np.vstack((lat_samples, lon_samples)))
    return location_samples


def get_random_image_sizes(city):
    n_sizes = city['n-images']
    mu, sigma = 350, 50
    lower, upper = 250, 450
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    sizes = np.round(X.rvs(n_sizes)).astype(int)
    return sizes


def get_google_maps_image(api_key, lat, lon, size, maptype):
    url = "https://maps.googleapis.com/maps/api/staticmap?" \
          "center=" + str(lat) + "," + str(lon) + \
          "&zoom=18" \
          "&size=" + str(size) + "x" + str(size) + \
          "&maptype=" + maptype + \
          "&key=" + api_key + \
          "&style=feature:all|element:labels|visibility:off" \
          "&scale=2"

    response = request.urlopen(url)
    image_pil = Image.open(io.BytesIO(response.read()))
    image_pil = image_pil.convert("RGB")
    image_BGR = np.asarray(image_pil)
    image_RGB = np.copy(image_BGR)
    image_RGB[:, :, 0] = image_BGR[:, :, 2]
    image_RGB[:, :, 2] = image_BGR[:, :, 0]
    return image_RGB


def process_images(satellite_image, road_image):
    height = np.size(satellite_image, 0)
    width = np.size(satellite_image, 1)
    satellite_image = satellite_image[0:height - 40, 40:, :]
    road_image = road_image[0:height - 40, 40:, :]
    road_image = cv2.resize(road_image, (400, 400), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("Original Road", road_image)

    road_image = cv2.cvtColor(road_image, cv2.COLOR_BGR2GRAY)
    _, road_image = cv2.threshold(road_image, 253, 255, type=cv2.THRESH_BINARY)

    is_road = road_image == 255
    percentage_road = np.sum(is_road.flatten()) / ((height - 40) * (width - 40))

    if percentage_road < 0.03:
        return False, None, None

    road_image = cv2.GaussianBlur(road_image, (13, 13), 5, borderType=cv2.BORDER_REFLECT101)
    _, road_image = cv2.threshold(road_image, 80, 255, type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(road_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    large_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            large_contours.append(contour)

    road_image_final = np.zeros((400, 400, 3))
    cv2.drawContours(road_image_final, large_contours, -1, (255, 255, 255), cv2.FILLED)
    satellite_image = cv2.resize(satellite_image, (400, 400), interpolation=cv2.INTER_AREA)

    # cv2.imshow("Satellite", satellite_image)
    # cv2.imshow("Road", road_image_final)
    # cv2.waitKey()

    return True, satellite_image, road_image_final


def get_one_data_pair(city, api_key, location, size):
    lat = location[0]
    lon = location[1]
    satellite_image = get_google_maps_image(api_key, lat, lon, size, maptype="satellite")
    road_image = get_google_maps_image(api_key, lat, lon, size, maptype="terrain")
    is_valid, satellite_image, road_image = process_images(satellite_image, road_image)
    while not is_valid:
        location = get_random_locations(city, 1).ravel()
        lat = location[0]
        lon = location[1]
        satellite_image = get_google_maps_image(api_key, lat, lon, size, maptype="satellite")
        road_image = get_google_maps_image(api_key, lat, lon, size, maptype="terrain")
        is_valid, satellite_image, road_image = process_images(satellite_image, road_image)

    return satellite_image, road_image


def save_images(city_name, satellite_image, road_image, index):
    filename = city_name + "-" + str(index).zfill(6) + ".png"
    cv2.imwrite("../data/google-maps-data/images/" + filename, satellite_image)
    cv2.imwrite("../data/google-maps-data/groundtruth/" + filename, road_image)

def find_file_saving_index(city_name):
    existing_images = sorted(glob.glob('../google-maps-data/images/' + city_name + '*'))

    if len(existing_images) == 0:
        return 0
    else:
        last_image_name = existing_images[len(existing_images) - 1]
        extension_index = last_image_name.rfind(".")
        last_image_index = int(last_image_name[extension_index - 6:extension_index])
        return last_image_index + 1


if __name__ == '__main__':
    google_api_key = "PLACE YOUR GOOGLE API KEY HERE"
    is_valid, filtered_data = input_check()
    if not is_valid:
        sys.exit(1)

    for city_name in filtered_data:
        city = filtered_data[city_name]
        locations = get_random_locations(city=city)
        image_sizes = get_random_image_sizes(city=city)
        start_index = find_file_saving_index(city_name)
        n_images = city["n-images"]
        for i in range(0, n_images):
            try: 
                image_size = image_sizes[i]
                location = locations[i]
                satellite_image, road_image = get_one_data_pair(city=city, api_key=google_api_key, location=location, size=image_size)
                save_images(city_name, satellite_image, road_image, start_index + i)
            except Exception as e:
                print(e)
                i = i-1
                continue
            if i % 100 == 0:
                print(city_name, i, '/', n_images)
