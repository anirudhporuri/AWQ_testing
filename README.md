# Quantization Benchmark Setup on Zaratan

## How to Run

### 1. SSH into Zaratan
```bash
ssh your_username@zaratan.umd.edu
```

### 2. Build and Install Python 3.11 Locally (with Headers)
```bash
cd ~
wget https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz
tar -xf Python-3.11.8.tgz
cd Python-3.11.8
./configure --prefix=$HOME/python311_with_dev --enable-optimizations
make -j4
make install
```

### 3. Create Virtual Environment
```bash
~/python311_with_dev/bin/python3.11 -m venv ~/bnb_quant
source ~/bnb_quant/bin/activate
```

### 4. Install Required Packages
```bash
pip install --upgrade pip setuptools wheel
pip install transformers bitsandbytes accelerate datasets evaluate tabulate
```

### 5. Add Python to PATH
```bash
echo 'export PATH=$HOME/python311_with_dev/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 6. Run Benchmark Script
```bash
sbatch quant.benchmark.slurm
```

### 7. Monitor Progress
```bash
squeue -u your_username
```
