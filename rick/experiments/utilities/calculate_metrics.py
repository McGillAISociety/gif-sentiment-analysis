import numpy as np


def calculate_accuracy(y_pred, y_true):
    a = y_pred.sigmoid().cpu().detach().numpy()
    a = np.where(a > 0.5, 1, 0).squeeze().astype('uint8')
    b = y_true.cpu().detach().numpy().squeeze().astype('uint8')

    correct = np.equal(a, b).astype('uint8')
    acuraccy = correct.sum() / len(a)
    return acuraccy
