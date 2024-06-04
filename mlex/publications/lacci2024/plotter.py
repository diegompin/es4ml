import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class Plotter():

    def plot_matrix(self, y_true, y_pred, name_cycler ,filename=None)->None: 
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        name = next(name_cycler)
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title( f"{name}" , fontsize=18)
        plt.tight_layout()
        if filename:
            plt.savefig(f"{filename}_{name}.pdf")
        else:
            plt.show()

        def plot_graphic(self, list_predictions)->None:
            sequence_length = 5

            title = "ROC"
            fpr, tpr, thresholds = metrics.roc_curve(y_test[sequence_length:-sequence_length+1], y_pred)
            auc = metrics.auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='deeppink', linewidth=4, label=f"ROC Curve (area = {round(auc,2) })")
            ax.plot([0,1], [0,1], "k--",linewidth=4, label='random classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
            ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
            ax.set_title(f"Receiver Operating Characteristic \n {title}", fontsize=18)
            ax.legend(loc="lower right")
            plt.savefig("../results/roc.pdf")
            plt.show()
