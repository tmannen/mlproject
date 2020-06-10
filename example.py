import h5py
from PIL import Image

data = h5py.File("empty_warehouse/_data/data0.hdf5", 'r')
print(data.keys())  # Print dataset groups below root
print(data["sensors"].keys())  # Print dataset groups below sensors
print(data["sensors"]["east"][0].shape)  # Print shape of first image from east-camera
vehicle_index = 0
print(data["state"]["location"][vehicle_index][5])  # Print vehicle location (x,y,z - in meters) on square 5

rows = 10
columns = 10

def index_to_xy(index):
    """ Dataset index to xy coordinates """
    x = index % columns
    y = index // columns
    return x, y

def xy_to_index(x, y):
    """ xy coordinates to dataset index """
    index = y * columns + x
    return index

def display_images(x, y):
    """ Display all images from given xy coordinates """
    index = xy_to_index(x, y)
    sensors = data['sensors']
    for direction in ["north", "east", "south", "west"]:
        north = sensors[direction][index]
        # Here Pillow-library is used for displaying images,
        # but you can use numpy or any other library to manipulate these images
        im = Image.fromarray(north)
        im.show()

def go_east(x, y):
    return x+1, y

def go_west(x, y):
    return x-1, y

def go_south(x,y):
    return x, y+1

def go_north(x,y):
    return x, y-1


""" Simple example for traversing the dataset and displaying images """
location = (0, 0)
location = go_east(*location)
location = go_south(*location)
display_images(*location)
