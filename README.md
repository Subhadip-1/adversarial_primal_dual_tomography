# adversarial_primal_dual_tomography
This repo contains a simple pytorch implementation of the adversarially learned primal-dual (ALPD) method for inverse problems, with applications to  CT reconstruction. Follow the steps below to run the code:

* Get the phantoms: `https://drive.google.com/drive/folders/1SHN-yti3MgLmmW_l0agZRzMVtp0kx6dD?usp=sharing`
* Create a conda environment with correct dependencies: `conda env create -f environment.yml`
* Check if `torch` got installed properly with GPU support, in which case `print(torch.cuda.is_available())` should show `True`. 
* Now activate your new conda environment by `conda activate env_deep_learning`, and install ODL from source (this step is important): 
  * `git clone https://github.com/odlgroup/odl`
  * `cd odl`
  * `pip install --editable .`
  * Now run `python`, then `import odl` and check if `print(odl.__version__)` outputs `1.0.0.dev0`. `odl 0.7.0` and `odl 1.0.0.dev0` compute different forward operators. This code runs with `odl 1.0.0.dev0`. 
* Simulate projection data: `python simulate_projections_for_train_and_test.py`. Should run properly if the downloaded directory named `mayo_data` is placed inside the cloned directory. Otherwise, modify `datapath` in the script appropriately.   
* Train and evaluate model (logs and networks are auto-saved): `python train_adversarial_LPD.py` 
 
