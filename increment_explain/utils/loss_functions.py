def mae_loss(y_true, y_prediction):
    return abs(y_true - y_prediction)


def mse_loss(y_true, y_prediction):
    return (y_true - y_prediction) ** 2
