# Code used in my undergrad thesis

Código usado para los experimentos descritos en mi tésis de pregrado titulada: "Algoritmo de clasificación de objetos en imágenes difractivas basado en medidas cuadráticas codificadas usando un enfoque de aprendizaje profundo"


Code implementation of the experiments used in my undergrad thesis titled: "Algoritmo de clasificación de objetos en imágenes difractivas basado en medidas cuadráticas codificadas usando un enfoque de aprendizaje profundo"


# Libraries used


* Python 3.9
* tensorflow-gpu 2.6
* matplotlib
* pandas
* opencv

# How to run

```{r, engine='bash', count_lines}
python main_classpr.py [dataset] config_files/config_forward_models.json 
```

the dataset parameter could be `mnist` or `fashion_mnist`

## Example:

python main_classpr.py mnist config_files/config_forward_models.json 


## Config file

The config file describe parameters of the propagation models used to simulate the coded diffraction patterns.


