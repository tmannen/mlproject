- jokaisella ilmansuunnalla oma kohina

- kokeilin super basic resnet18 jossa yrittää ennustaa oikean squaren, koodi melkein suoraan täältä: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html . ei augmentointia
- vain 800 imagea, jaettu train, val, test. 24 epochia. val accuracy nousee 0.35 about. train accuracy melkein 1.00, eli overfittaa?

## 17.06.2020

- First multi view attempt: take all 4 images, take feature vector and concatenate. classify using these.