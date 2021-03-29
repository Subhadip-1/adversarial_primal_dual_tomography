# adversarial_primal_dual_tomography
Follow the steps below:

* Get the phantoms: 
* Create a conda environment with correct dependencies: `conda env create -f environment.yml`
* Check if `torch` got installed properly with GPU support, in which case `print(torch.cuda.is_available())` should be `True`. 
* Now activate your new conda environment, and install ODL from source (this step is important): 
  * `git clone https://github.com/odlgroup/odl`
  * `cd odl`
  * `pip install --editable .`
  * Now run `python`, then `import odl` and check if `print(odl.__version__)` outputs `1.0.0.dev0`. `odl 0.7.0` and `odl 1.0.0.dev0` compute different forward operators. This code runs with `odl 1.0.0.dev0`. 
