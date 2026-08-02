![LOGO](asset/logo.png)

Reproducible material for **PINNslope: seismic data interpolation and local slope estimation with physics informed neural networks - Brandolin F., Ravasi M., Alkhalifah  T.**


## Project structure
This repository is organized as follows:

* :open_file_folder: **pinnslope**: python library containing routines for "PINNslope" seismic data interpolation and local slope estimation with physics informed neural networks;
* :open_file_folder: **data**: folder containing input data and results;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
* :open_file_folder: **asset**: folder containing logo;

## Notebooks
The following notebooks are provided:

- :orange_book: ``PINNslopePE.ipynb`` : notebook performing field seismic data interpolation and local slope estimation.
- :orange_book: ``PINNslope_synth.ipynb`` : notebook performing synthetic seismic data interpolation and local slope estimation
- :orange_book: ``LS_PWreg_Inversion.ipynb`` : notebook performing plane-wave regularized least-squares interpolation.
- :orange_book: ``plottingREALD.ipynb`` : notebook reproducing the figures in the paper (of the field data numerical examples).
- :orange_book: ``plottingSYNTH.ipynb`` : notebook reproducing the figures in the paper (of the synth data numerical examples).  

## Dual-angle interpolation

The `feature/dual-angle` implementation adds a double-output wavefield network,
a double-output slope network, constrained positive/negative slope ranges, and
the two plane-wave physics losses used for signal separation. See
`docs/DUAL_ANGLE_DESIGN.md` for the model equations and run the CPU-friendly
example with:

```powershell
python -m examples.dual_angle_interpolation --epochs 200 --device cpu
```

Run the checkpointed synthetic interpolation baseline on CUDA with:

```bash
python -m examples.train_dual_angle \
  --data data/synth_data/data_dense.npy \
  --output-dir runs/synth_dual_angle \
  --epochs 2000 \
  --device cuda
```

The default crop, sampling interval, and every-fifth-trace restriction match
the synthetic paper notebook. The experiment writes resumable checkpoints,
epoch losses, both separated wavefields and slopes, the reconstruction, and
observed/missing-trace metrics under the output directory.

Render the exported arrays as PNG figures with:

```bash
python -m examples.plot_dual_angle_results \
  --run-dir runs/synth_dual_angle_2000
```

## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate envpinnslope
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite Us

Brandolin, F., Ravasi, M., & Alkhalifah, T. (2024). Pinnslope: Seismic data interpolation and local slope estimation with physics informed neural networks. GEOPHYSICS , 89 (4), V331-V345. [DOI: 10.1190/geo2023-0323.1](https://doi.org/10.1190/geo2023-0323.1)

Please use the following BibTeX entry to cite this work:

```bibtex
@article{doi:10.1190/geo2023-0323.1,
author = {Francesco Brandolin and Matteo Ravasi and Tariq Alkhalifah},
title = {PINNslope: Seismic data interpolation and local slope estimation with physics informed neural networks},
journal = {GEOPHYSICS},
volume = {89},
number = {4},
pages = {V331-V345},
year = {2024},
doi = {10.1190/geo2023-0323.1},
URL = { https://doi.org/10.1190/geo2023-0323.1},
eprint = {https://doi.org/10.1190/geo2023-0323.1},

}
