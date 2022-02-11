import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"

    #datasets_name = ["fashion_mnist", "mnist"]
    num_classes = 10
    model_types = ["none", "back", "fsi"]
    classifiers = ["mobilnet", "xception", "inception"]
    forward_models = ["ASM", "FRESNEL", "FRAN"]
    datasets = ["mnist", "fashion_mnist"]
    metrics = ["accuracy", "f1_score"]
    results_folder_root = "results_kfold"
    ext = ".svg"

    
    results_total = np.zeros((len(metrics), len(datasets), len(classifiers), len(model_types), len(forward_models)))
    for indx_datasets, dataset in enumerate(datasets):
        for indx_classifiers, clasification_network in enumerate(classifiers):
            for indx_model_types, model_type in enumerate(model_types):
                for indx_forward_models, forward_model in enumerate(forward_models):
                    results_path =  os.path.join(results_folder_root, forward_model,  model_type, clasification_network, dataset, "test.csv")
                    results = pd.read_csv(results_path)
                    for indx_metrics, metric in enumerate(metrics):
                        results_total[indx_metrics, indx_datasets, indx_classifiers, indx_model_types, indx_forward_models] = results[metric].values[0]
                        #print("##########################################")
    results_total = np.random.rand(len(metrics), len(datasets), len(classifiers), len(model_types), len(forward_models))
    for indx_metrics, metric in enumerate(metrics):
        #fig = plt.figure(figsize=(30, 10))
        fig, axs = plt.subplots(len(datasets), len(classifiers), constrained_layout=False, sharey=False, sharex = False, figsize=(20, 10))
        fig.suptitle(metric.capitalize(), fontsize=16)
        for indx_datasets, dataset in enumerate(datasets):
            for indx_classifiers, clasification_network in enumerate(classifiers):

                    im = axs[indx_datasets, indx_classifiers].imshow(results_total[indx_metrics, indx_datasets, indx_classifiers, ...], cmap="gray", vmin=0, vmax=1)
                    axs[indx_datasets, indx_classifiers].set_title(clasification_network.capitalize())
                    #axs[indx_datasets, indx_classifiers].set_xlabel('Forward Model')
                    #axs[indx_datasets, indx_classifiers].set_ylabel('Model Type')
                    axs[indx_datasets, indx_classifiers].set_xticks(np.arange(len(forward_models)))
                    axs[indx_datasets, indx_classifiers].set_yticks(np.arange(len(model_types)))
                    # ... and label them with the respective list entries
                    axs[indx_datasets, indx_classifiers].set_xticklabels([s.capitalize() for s in forward_models])
                    axs[indx_datasets, indx_classifiers].set_yticklabels([s.capitalize() for s in model_types])

                    # axs[indx_datasets, indx_classifiers].set_xticks(np.arange(len(forward_models)), labels=forward_models)
                    # axs[indx_datasets, indx_classifiers].set_yticks(np.arange(len(model_types)), labels=model_types)

        fig.colorbar(im, ax=axs.ravel().tolist())
        #fig.clim(0, 1)
        #fig.tight_layout()
        #plt.draw()
        plt.savefig(os.path.join(results_folder_root, "test_result_" + metric + ext))
    #plt.show()

    
