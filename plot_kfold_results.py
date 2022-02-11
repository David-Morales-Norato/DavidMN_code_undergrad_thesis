import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    #datasets_name = ["fashion_mnist", "mnist"]
    num_classes = 10
    model_types = ["none", "back", "fsi"]
    classifiers = ["mobilnet"]
    forward_models = ["ASM", "FRESNEL", "FRAN"]
    datasets = ["fashion_mnist"]
    metrics = ["accuracy", "f1_score"]
    results_folder_root = "results_kfold"
    model_table_path = os.path.join("utils", "model_table.tgn")
    kfolds = 10
    

    results_total = np.zeros((kfolds, len(metrics), len(datasets), len(classifiers), len(model_types), len(forward_models)))
    for indx_datasets, dataset in enumerate(datasets):
        for indx_classifiers, clasification_network in enumerate(classifiers):
            for indx_model_types, model_type in enumerate(model_types):
                for indx_forward_models, forward_model in enumerate(forward_models):
                    for k_fold in range(kfolds):
                        results_path =  os.path.join(results_folder_root, forward_model,  model_type, clasification_network, dataset, "fold_"+str(k_fold), "test.csv")
                        results = pd.read_csv(results_path)
                        for indx_metrics, metric in enumerate(metrics):
                            results_total[k_fold, indx_metrics, indx_datasets, indx_classifiers, indx_model_types, indx_forward_models] = results[metric].values[0]
                            #print("##########################################")

    #
    results_total = np.random.rand(kfolds, len(metrics), len(datasets), len(classifiers), len(model_types), len(forward_models))
    mean_results = results_total.mean(axis = 0)
    stdv_results = results_total.std(axis = 0)


    with open(model_table_path, "r") as file:
        line = file.readline()


    for indx_datasets, dataset in enumerate(datasets):
        tabla = line
        cont = 1
        for indx_classifiers, clasification_network in enumerate(classifiers):
            for indx_metrics, metric in enumerate(metrics):
                for indx_forward_models, forward_model in enumerate(forward_models):
                    for indx_model_types, model_type in enumerate(model_types):
                        mean = mean_results[indx_metrics, indx_datasets, indx_classifiers, indx_model_types, indx_forward_models]
                        std = stdv_results[indx_metrics, indx_datasets, indx_classifiers, indx_model_types, indx_forward_models]
                        #tabla = tabla.replace("\"r_"+str(cont) + "\"","\""+"$" + str(round(mean,2)) + " pm " + str(round(std,2))+ "$" +"\"" )
                        tabla = tabla.replace("\"r_"+str(cont) + "\"","\"" + model_type+"_"+forward_model+"_"+metric+clasification_network+"_"+dataset + "\"")
                        
                        cont = cont +1

        print("CONT", cont)
        with open(os.path.join(results_folder_root, "tabla_" +dataset+".tgn"), "w") as file:
            file.write(tabla)






