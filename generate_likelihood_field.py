import cv2
import numpy as np

#read in the pgm map
map_image = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
#Set vales based on color scale in map
_, binary_map = cv2.threshold(map_image, 250, 255, cv2.THRESH_BINARY)
#Calculate the distance to closest wall
likelihood_field = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
#Set a max of 15
max_distance = 10000
likelihood_field[likelihood_field > max_distance] = max_distance
likelihood_field_norm = cv2.normalize(likelihood_field, None, 0, 255, cv2.NORM_MINMAX)
#Save the file as a npy file
np.save('likelihood_field.npy', likelihood_field)
#Write the file to a png
cv2.imwrite('likelihood_field.png', likelihood_field_norm)

