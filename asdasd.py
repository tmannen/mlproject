import models
import create_dataset

mvc = models.MVCNN()

image_datasets = create_dataset.create_dataset(True)
t = image_datasets['test']
mvc(t[0][0].unsqueeze(0))

