#!/bin/bash

echo Upload dataset

kaggle datasets version -p trainimg -m "Updated data"

echo Upload notebook

# notebook="michaelnnka/catheter-inference"
# folder="./kaggle/inference"

notebook="michaelnnka/catheter-training"
folder="./kaggle/training_v1"
# folder="./kaggle/cv"

folder_to_remove="${folder}/*"
rm $folder_to_remove

kaggle kernels pull $notebook -p $folder -m
kaggle kernels push -p $folder