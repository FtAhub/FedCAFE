## FedCAFE
PyTorch implementation of FedCAFE


## Description
This is a simplified demo for the paper: FedCAFE: Federated Cross-Modal Hashing with Adaptive Feature Enhancement, including:
`img_module.py`: network architecture for image modality.
`txt_module.py`: network architecture for text modality.
`data_handler.py`: function for loading and splitting data.
`fed_utils.py`: function for weight aggregation.
`fed_task.py`: function for simulating federated communication.
`options.py`: hyperparameter settings.
`update.py`: function for local updates.
`utils.py`: tools for evaluating model performance.
`main.py`: a demo for our method.

## Version
- Python 3.7
- Pytorch 1.13.1

As Openreview does not allow the uploading of files which are larger than 100MB, the trained model and dataset are not available in our supplementary materials.

Please download the dataset from the Internet as all chosen datasets are publicly available.