from sklearn.metrics import (precision_recall_curve,
                             average_precision_score,
                            PrecisionRecallDisplay,f1_score,confusion_matrix)
from sklearn.model_selection import (cross_val_score,cross_validate,
                                    cross_val_predict)
from sklearn.preprocessing import (StandardScaler,RobustScaler,
                                PolynomialFeatures,OrdinalEncoder,
                                   FunctionTransformer,OneHotEncoder,
                                   LabelBinarizer)
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
def get_results(model,X,y):
    '''
    Return PR curve and confusion matrix for each class
    model: model instance
    X: array-like with shape (n_samples,n_features)
    y: array-like ordinal categories with shape (n_samples,)
    '''
    #make predictions for model
    y_hat = cross_val_predict(model,X,y,method="predict_proba")
    predictions = cross_val_predict(model,X,y)
    #calculate f1 score 
    f1 = f1_score(y_true=y,y_pred=predictions,average="micro")
    #build confusion matrix
    matrix = confusion_matrix(y,predictions)

    
    binarizer = LabelBinarizer()
    y =  binarizer.fit_transform(y)
    n_classes = y.shape[1]
    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    #create fig and axes
    _, ax = plt.subplots(ncols=2,figsize=(12, 7))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i],recall[i],_ = precision_recall_curve(y[:,i],y_hat[:,i])
        average_precision[i] = average_precision_score(y[:,i],y_hat[:,i])
    precision["micro"],recall["micro"],_ = precision_recall_curve(
                                        y.ravel(),y_hat.ravel()
                                        )
    average_precision["micro"] = average_precision_score(y,
                                                         y_hat,
                                                         average="micro")

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax[0].plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax[0].annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax[0], name="Micro-average precision-recall",
                 color="red",
                 linestyle="dashed")
    classes = ["big","low","medium","tank"]
    #plot every precision-recall curve in ax
    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax[0], name=f"Precision-recall for class {classes[i]}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].legend(handles=handles, labels=labels, loc="best")
    ax[0].set_title("Precision-Recall")
    #plot confusion matrix and edit labels
    ax[1].matshow(matrix,cmap=plt.cm.cividis)
    ax[1].set_ylabel("Truth")
    ax[1].set_xlabel("Predictied")
    ax[1].set_title("Confusion matrix")
    plt.show()
    print(f"F1 score(average): {f1}")

def plot_f1(models):
    '''
    returns a bar plot of f1 scores
    models: dict type, name and model instance
    '''
    f1_scores = dict()
    for name,model in models.items():
        temp_prediction = cross_val_predict(model,df_train,y_train,method="predict")
        temp_f1 = f1_score(y_true=y_train,y_pred=temp_prediction,average="micro")
        f1_scores[name] = temp_f1
    fig,ax = plt.subplots(figsize=(7,5))
    ax.bar(x=list(f1_scores.keys()),height=list(f1_scores.values()))
    ax.set_title("F1 scores")
    plt.show()
