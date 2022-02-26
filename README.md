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


To save the working environment with all packages and versions so others can retrieve and run on the trained models without compatibility issue:
```
conda env export > environment.yml
```
In another machine, edit the yml file to change the env name on line 1 and prefix on last line, then run without any conda environment:
```
conda env create -f environment.yml
```
If error "ResolvePackageNotFound" appears, edit the yml file to move the packages under the pip section.
