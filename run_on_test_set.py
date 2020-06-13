import torch
import models
import create_dataset


if __name__ == '__main__': 
    device = 'cuda:0'

    model = models.create_basic_model(device)
    model.load_state_dict(torch.load("models/basic_model.pt"))
    model.eval()

    data = create_dataset.create_dataset()
    image_datasets = create_dataset.create_dataset()

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                shuffle=True, num_workers=1)
                for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    preds = []
    gts = []

    for im, trgt in dataloaders['test']:
        pred = torch.argmax(model(im.to(device)))

        preds.append(pred.to('cpu'))
        gts.append(trgt)


    preds = torch.Tensor(preds)
    gts = torch.Tensor(gts)
    correct = preds == gts
    correct_sum = torch.sum(correct).to(dtype=torch.float)

    print("%f of datapoints were accurately predicted." % (correct_sum/len(preds)).item())