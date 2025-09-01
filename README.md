This repository contains the official implementation for the paper **FlareDTDG: Harnessing Temporal Recency for Scalable Discrete-Time Dynamic Graph Training**.

## Requirements

To run the experiments, you will need the following:

*   **Hardware**:
    *   A machine with at least 4 NVIDIA GPUs.
    *   Sufficient RAM and disk space for datasets and processed files.
*   **Software**:
    *   Linux-based OS
    *   NVIDIA drivers and CUDA Toolkit
    *   Miniconda or Anaconda
    *   Python 3.8+

## 1. Installation

First, clone the repository and set up the Python environment.

```bash
# Clone this repository
git clone https://github.com/wjie98/FlareDTDG
cd FlareDTDG

# Create and activate a conda environment
conda create -n sgl python=3.10
conda activate sgl

# Install dependencies
pip install -r requirements.txt
```

## 2. Dataset Preparation

The scripts expect datasets to be pre-processed into a specific `.pth` format.

1.  **Download Datasets**:
    Download the raw graph data from the [Network Repository](https://networkrepository.com/dynamic.php).

2.  **Pre-process and Place Data**:
    You must first convert the raw dataset files (e.g., `.csv`, `.txt`) into the format expected by our scripts located in `./flare/utils/web_data.py`.

    After pre-processing, place the final `.pth` files in the following directory. You may need to create it first.

    ```bash
    mkdir -p ~/DATA/FlareGraph/web
    mv your_dataset.pth ~/DATA/FlareGraph/web/
    ```
    For example, the Slashdot dataset should be located at `~/DATA/FlareGraph/web/ia-slashdot-reply-dir.pth`.

## 3. Reproduction Pipeline

Follow these steps to partition the data and run the training.

### Step 1: Two-Level Graph Partitioning

This script partitions the graph for distributed training. It first divides the graph into a number of partitions equal to the number of GPUs (`num_inter_parts`). Then, it further subdivides each partition into smaller chunks (`num_intra_parts`) for sequential processing within our temporal model.

*   **Action**: Run the `prepare_nparts.py` script.
*   **Configuration**: The number of inter- and intra-partitions are set inside the script's `if __name__ == "__main__":` block. The default is set for 4 GPUs.
    *   `num_inter_parts = 4`
    *   `num_intra_parts = 128`
*   **Command**:
    ```bash
    python prepare_nparts.py
    ```
*   **Output**: Partition assignment files will be saved to `~/DATA/FlareGraph/nparts/`.

### Step 2: Prepare Partitioned Data for Each GPU

This script uses the partition assignments from the previous step to split the node features and graph structure into separate files, one for each GPU process. It also computes necessary metadata, such as GCN normalization factors and communication indices for cross-partition edges.

*   **Action**: Run the `prepare_part_data.py` script.
*   **Configuration**: The `params` dictionary within the script maps the total number of GPUs (`k`) to the partitioning scheme (`num_inter_parts`, `num_intra_parts`). Ensure the entry for `k=4` matches the parameters from Step 1.
*   **Command**:
    ```bash
    python prepare_part_data.py
    ```
*   **Output**: Processed data for each partition will be saved in `~/DATA/FlareGraph/processed/[dataset_name]/`. For example, `~/DATA/FlareGraph/processed/ia-slashdot-reply-dir/004_001.pth` is the data for GPU 0 when training with 4 GPUs.

### Step 3: Run Distributed Training

The provided `run_flare.sh` script is the easiest way to launch the distributed training job. It sets the `OMP_NUM_THREADS` environment variable and uses `torchrun` to spawn 4 processes.

*   **Action**: Execute the shell script.
*   **Configuration**: The script is pre-configured to run the `mpnn_lstm` model on the `ia-slashdot-reply-dir` dataset using 4 GPUs. You can edit the `MODEL_CHOICES` and `DATA_CHOICES` arrays in the script to run other experiments.
*   **Command**:
    ```bash
    bash run_flare.sh
    ```
*   **Output**: Logs and results, including training/evaluation loss per epoch, will be saved to the `./logs_flare` directory.

## Running on a Different Number of GPUs

To scale the experiment to a different number of GPUs (e.g., 8), you must adjust the parameters in the preprocessing scripts to match the new world size.

Let's assume you want to run on **8 GPUs**:

1.  **In `prepare_nparts.py`**:
    Update `num_inter_parts` to match the GPU count. You may also want to adjust `num_intra_parts`.
    ```python
    num_inter_parts = 8
    num_intra_parts = 64 # Example value
    ```
    Then, run `python prepare_nparts.py`.

2.  **In `prepare_part_data.py`**:
    Add a new entry to the `params` dictionary where the key is the number of GPUs. The values must correspond to the parameters set in the previous step.
    ```python
    params = {
        # ... existing entries
        8: (8, 64), # (num_inter_parts, num_intra_parts)
    }
    ```
    Then, run `python prepare_part_data.py`.

3.  **In `run_flare.sh`**:
    Change the `--nproc_per_node` argument for `torchrun` to the new GPU count.
    ```bash
    torchrun \
      --nproc_per_node 8 \
      # ... rest of the command
    ```
    Then, execute `bash run_flare.sh`.
