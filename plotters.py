import matplotlib.pyplot as plt


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()
    plt.xlim(-50000, 50000)
    plt.ylim(0, 1)
    plt.grid()
    #plt.axvline(x=8000, color="r")
    #plt.axvline(x=np.interp(8000, recalls)color='grey')
    plt.xlabel('Threshold', fontsize=11)


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, label=None)
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)


def plot_roc_curve(fpr, tpr, label='ROC'):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.grid()
    plt.legend()# Add axis labels and grid
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('Recall/TPR', fontsize=11)


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")