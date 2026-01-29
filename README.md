# LLM-BFFD
## Project Overview
This repository provides the implementation of a fault diagnosis model for the Blast Furnace Ironmaking Process (BFIP) using large language models (LLMs). The process involves loading a pre-trained model, performing fine-tuning with a customized dataset, and evaluating the model's performance. Below is a step-by-step explanation of the implementation.
## Project Structure
Here is the directory structure of the project:
```
blast-furnace-fault-diagnosis/
├── LLM-BFFD.ipynb          # Jupyter notebook containing all code and analysis
├── data/                   # Data folder containing training and test datasets
│   ├── traindataset.jsonl         # Training dataset
│   └── testdataset.jsonl          # Test dataset
├── models/                 # Folder to store pretrained and fine-tuned models
│   └── pretrained/         # Pretrained models
├── README.md               # Project documentation
└── outputs/                # Folder to store output files, such as models and logs

```
## Data Preparation
The dataset should be in .jsonl format (JSON Lines), where each line represents a single sample. You will need the following data files:
- **traindataset.jsonl**: Contains the training dataset.
- **testdataset.jsonl**: Contains the test dataset.
  
The training and test datasets used in this project are derived from real-world data collected during the blast furnace ironmaking process. Due to concerns regarding corporate confidentiality and privacy, the original datasets cannot be publicly provided.

## Experiment Steps
1. Loading the Pre-trained Model
The first step is to deploy a lightweight model from Hugging Face to the local machine, with the model being loaded from the local cache directory `./model_cache`.
2. Pre-Fine-tuning Testing
Before fine-tuning the model, an initial test is conducted using a predefined prompt structure. The input prompt provides a task description for the fault diagnosis of BFIP, and the model is asked to output fault categories based on the input. The output is returned in JSON format, containing the reasoning and the predicted fault label.
3. Loading the Dataset
The dataset for fine-tuning is loaded from a JSONL file. The dataset is processed to create training prompts that align with the BFIP fault diagnosis task. The formatting function processes the input, reasoning, and output fields to match the prompt structure required for training.
4. Fine-tuning the Model
The model's low-rank adaptation (LoRA) method is used for parameter-efficient fine-tuning. The training configuration includes hyperparameters like batch size, gradient accumulation steps, warmup steps, and learning rate. The training is conducted for 30 epochs. 
5. Model Evaluation
After fine-tuning, the model is evaluated using a test dataset. The evaluation involves generating predictions based on test samples, comparing the model's output with the ground truth, and calculating various performance metrics such as accuracy, precision, recall, and F1-score. Confusion matrices are also generated to visually inspect the model's performance across the fault categories.
