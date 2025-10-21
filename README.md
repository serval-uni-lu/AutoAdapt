# AutoAdapt — On the Application of AutoML for Parameter-Efficient Fine-Tuning of Pre-Trained Code Models

This repository provides the full source code and scripts required to replicate the experiments reported in the *AutoAdapt* research paper.

The evaluated software engineering tasks include:

- **Code search** (`main_codeSearch.py`)  
  - Dataset: **AdvTest** 
  - Data: https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-Adv

- **Defect detection** (`main_defect.py`)  
  - Dataset: **Devign** 
  - CodeXGLUE page: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection  
  - Large JSON download (mirror): https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view

- **Clone detection** (`main_clone.py`)  
  - Dataset: **BigCloneBench** 
  - General access / references: https://github.com/github/CodeSearchNet

The optimization logic (evolutionary adapter-architecture search) is implemented in `optimization.py`.

---

## Prerequisites

- Python 3.8+  
- PyTorch (GPU recommended)  
- `transformers`, `numpy`, `scikit-learn`, `tqdm`, `datasets` (as needed), etc.


### Local `myOpenDelta` (editable install)

This repo includes a modified local implementation of OpenDelta used to insert custom adapter architectures. Install it in editable mode from the repository root:

```bash
pip install -e ./myOpenDelta
```

## Running 

Use the provided *.sh runner scripts or call the Python entry points directly.

## Citation

Please cite this work using the following:

```bash
@article{10.1145/3734867,
  author = {Akli, Amal and Cordy, Maxime and Papadakis, Mike and Le Traon, Yves},
  title = {AutoAdapt: On the Application of AutoML for Parameter-Efficient Fine-Tuning of Pre-Trained Code Models},
  year = {2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  issn = {1049-331X},
  url = {https://doi.org/10.1145/3734867},
  doi = {10.1145/3734867},
  abstract = {},
  note = {Just Accepted},
  journal = {ACM Trans. Softw. Eng. Methodol.},
  month = may,
  keywords = {PEFT, pre-trained code models, Optimization, AutoML, NAS}
}
```

