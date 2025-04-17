'''
This script updates any new observations into the combined dataset 
stored on Hugging Face Hub.

If the operation fails, it exits with non-zero status so the workflow
fails too (better observability of the failures)
'''

import os 
from huggingface_hub import HfApi
from src.dataset_handling import sync_dataset

############################################################
dataset_id = "Saving-Willy/temp_dataset"
token = os.getenv("HF_TOKEN")
dataset_filename = "data/train-00000-of-00001.parquet"
############################################################

if __name__ == '__main__':
    # Initialize API client
    api = HfApi(token=token)

    # add json files to dataset
    n_new = sync_dataset(
        api, dataset_id, dataset_filename, 
        create_dataset_if_not_exists=False)

    if n_new is None:
        # something went wrong. - want the workflow to fail so nonzero exit  
        print(f"Failed to sync dataset {dataset_id}.")
        raise SystemExit(1) 
