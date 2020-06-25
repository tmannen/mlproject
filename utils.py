import matplotlib.pyplot as plt

def index_to_xy(index):
    """ Dataset index to xy coordinates """
    x = index % columns
    y = index // columns
    return x, y

def xy_to_index(x, y):
    """ xy coordinates to dataset index """
    index = y * columns + x
    return index

def display_images(data, x, y):
    index = xy_to_index(x, y)
    sensors = data['sensors']

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(sensors['north'][index])
    axarr[0,1].imshow(sensors['east'][index])
    axarr[1,0].imshow(sensors['south'][index])
    axarr[1,1].imshow(sensors['west'][index])
    plt.show()

def display_dataset_images(datapoint):
    f, axarr = plt.subplots(2,2)
    directions = ['north', 'east', 'south', 'west']
    axes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    print("Label: ", datapoint[1])
    for ilmansuunta, ax, img in zip(directions, axes, datapoint[0]):
        axarr[ax[0],ax[1]].imshow(img)

    plt.show()