# haabbtt_neural_network

Neural network trained for h->aa->bbtautau signal/background separation

Using conda, create a virtual environment with tensorflow and other packages installed:
```
conda create -n tf tensorflow
conda activate tf
conda install root -c conda-forge
...
```

- Prepare ntuple files in root_raw/ as inputs
- Write outputs to new ntuple files in root_output/
- Model training and output writing are done in different notebooks
- Models are trained separately for different categories (1 bjet and 2bjets), all eras mixed, so two notebooks per channel
- Trained models as well as standardizing parameters (scaler) are saved in folder
- Output writing (nominal and shape uncertainties) is done in one notebook per channel
