'''
This script builds a new dataset from scratch, from all the
observations stored in the huggingface hub.

It is intended to be used rarely, e.g. when the data format
has changed, and we need to rebuild the dataset from scratch;
and so we don't expect a cron/continuous action triggering it.
'''

import os 
from huggingface_hub import HfApi
from src.dataset_handling import reset_dataset_rebuild_from_json

############################################################
dataset_ids = ["Saving-Willy/temp_dataset", "Saving-Willy/main_dataset"]
token = os.getenv("HF_TOKEN")
############################################################

if __name__ == '__main__':
    # Initialize API client
    api = HfApi(token=token)

    for dataset_id in dataset_ids:
        # run the full rebuild/reset, and push to hub
        reset_dataset_rebuild_from_json(
            api, dataset_id)
    