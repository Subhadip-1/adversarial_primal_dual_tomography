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

* Get the phantoms here: https://drive.google.com/drive/folders/1SHN-yti3MgLmmW_l0agZRzMVtp0kx6dD?usp=sharing. This will download a `.zip` file which you have to unzip.
* Create a conda environment with correct dependencies: `conda env create -f environment.yml`
* Check if `torch` got installed properly with GPU support, in which case `print(torch.cuda.is_available())` should show `True`. 
* Simulate projection data: `python simulate_projections_for_train_and_test.py`. Should run properly if the downloaded directory named `mayo_data` is placed inside the cloned directory. Otherwise, modify `datapath` in the script appropriately.   
* Train and evaluate model (logs and networks are auto-saved): `python train_adversarial_LPD.py`
* You can download the pre-trained LPD-based generator from https://drive.google.com/file/d/1GR4yeHcCBkvoUKxRcHIyD9ssKU7lnZry/view?usp=sharing and put it inside a sub-folder named `trained_models`. Once downloaded, run `python eval_adversarial_LPD.py` to run the model on test slices. Appropriately modify the filenames of the saved models in the eval script.  
 
