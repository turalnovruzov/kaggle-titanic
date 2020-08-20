import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def param_curve(model, param, X, y, start=1, end=100, step=1, verbose=True):
    x_ax = []
    scores = []
    val_scores = []

    for n in np.arange(start, end, step):
        model.set_params(**{param: n})
        model.fit(X, y)
        x_ax.append(n)
        scores.append(model.score(X, y))
        val_scores.append(cross_val_score(model, X, y).mean())
        if verbose:
            print(f'n = {n}\tscore = {scores[-1]}\tval score = {val_scores[-1]}')
    
    plt.plot(x_ax, scores, label='Train scores')
    plt.plot(x_ax, val_scores, label='Validation scores')
    plt.xlabel(param)
    plt.ylabel('Cost')
    plt.show()