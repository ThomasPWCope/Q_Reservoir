# Quantum Reservoir Computing Surrogate

Surrogate: A 'quantum neural network' imitation of the Siemens reactor.

Quantum Reservoir Computing: A fixed quantum circuit and a trained classical linear layer.

Training (fitting) is done with a quadratic loss and ordinary least squares (OLS).

Contains three main QRC methods:
- QRewindingStatevectorRC (works well)
- QExtremeLearningMachine (works but no quantum advantage)
- QContinuousRC (not working yet)

And three main tasks:
- Surrogate from Siemens reactor data (QLindA)
- Nonlinear autoregressive moving average (NARMA) 
- Short term memory (STM)

## Quickstart

How to use the code: [./notebooks/try_me.ipynb](./notebooks/try_me.ipynb)

Plots for the QLindA report: [./experiments/plotting/plot_rewinding_reactor.ipynb](./experiments/plotting/plot_rewinding_reactor.ipynb)

`/notebooks/` examples for benchmark tasks and QRC methods

`/experiments/plotting/` plot results from experiments

`/src/` contains the QRC implementation

## Documentation

Open [./src/docs/build/html/index.html](./src/docs/build/html/index.html) in your browser

or `<your_code_folder>/quantum-reinforcement-learning/qrc_surrogate/src/docs/build/html/index.html`

## Setup: Conda (MacOS)

```bash
conda env create --name envrl --file=env_rl.yml

conda activate envrl
```

## Setup: Conda (Windows)

```bash
conda create --name "envrl" python=3.11
conda activate envrl

conda install jupyter notebook matplotlib ipykernel seaborn -y
conda install -c anaconda ipython statsmodels scikit-learn pytables sphinx -y
conda install -c conda-forge ipympl gymnasium tqdm fastparquet sphinx-rtd-theme readthedocs-sphinx-ext sphinxcontrib-napoleon -y
conda install -c pytorch pytorch -y

conda install pip
# python -m pip install qiskit 'qiskit[providers]' # for MacOS
python -m pip install qiskit qiskit[providers]
python -m pip install qiskit_ibm_provider
# python -m pip install 'qiskit[visualization]' # for MacOS
python -m pip install qiskit[visualization]
```

---

## What I did (for reproducability)

### Conda 

Export the conda environment

```bash
conda env export | grep -v "^prefix: " > env_rl.yml
```

### Documentation (Sphinx)

```bash
mkdir docs
cd docs

sphinx-quickstart --ext-autodoc
```

Separate source and build folders: yes

Edit conf.py and index.rst

Redo this if you change the comments in the code:

```bash
conda activate envrl
sphinx-apidoc -f -o ../docs/source/ ../src/
make clean
make html
```
