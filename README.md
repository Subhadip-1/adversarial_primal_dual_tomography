# adversarial_primal_dual_tomography
This repo contains a simple pytorch implementation of the adversarially learned primal-dual (ALPD) method for inverse problems, with applications to  CT reconstruction. See the paper at https://arxiv.org/abs/2103.16151 for a detailed explanation of the algorithm.  

If you're using the scripts for your research, please conside citing the paper: 

```
@misc{mukherjee2021adversarially,
      title={Adversarially learned iterative reconstruction for imaging inverse problems}, 
      author={Subhadip Mukherjee and Ozan Öktem and Carola-Bibiane Schönlieb},
      year={2021},
      eprint={2103.16151},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

# Follow the steps below to run the code:

* Get the phantoms here: https://drive.google.com/drive/folders/1SHN-yti3MgLmmW_l0agZRzMVtp0kx6dD?usp=sharing.
* Create a conda environment with correct dependencies: `conda env create -f environment.yml`
* Check if `torch` got installed properly with GPU support, in which case `print(torch.cuda.is_available())` should show `True`. 
* Now activate your new conda environment by `conda activate env_deep_learning`, and install ODL from source (this step is important): 
  * `git clone https://github.com/odlgroup/odl`
  * `cd odl`
  * `pip install --editable .`
  * Now run `python`, then `import odl` and check if `print(odl.__version__)` outputs `1.0.0.dev0`. `odl 0.7.0` and `odl 1.0.0.dev0` compute different forward operators. This code runs with `odl 1.0.0.dev0`. 
* Simulate projection data: `python simulate_projections_for_train_and_test.py`. Should run properly if the downloaded directory named `mayo_data` is placed inside the cloned directory. Otherwise, modify `datapath` in the script appropriately.   
* Train and evaluate model (logs and networks are auto-saved): `python train_adversarial_LPD.py`
* You can download the pre-trained LPD-based generator from `https://drive.google.com/file/d/1GR4yeHcCBkvoUKxRcHIyD9ssKU7lnZry/view?usp=sharing` and put in inside a sub-folder named `trained_models`. Once downloaded, run `python eval_adversarial_LPD.py` to run the model on test slices.  
 
