# HuggingFace Timm model benchmarks

## Introduction
The [pytorch-image-models](https://github.com/huggingface/pytorch-image-models/tree/main/results) hosts the benchmarks of different models on the ImageNet dataset. However, these results are missing the "mdoel size" metric, which helps in making comparisons. Model_size is especially better for the following reasons,
1. **Resource Constraints**
   * Model size directly reflects the memory footprint a model occupies on a device. This is critical, as production environments often have limited memory resources, especially on edge devices like smartphones or embedded systems.
2. **Architectural Efficiency**
    *  Model architectures with similar parameter counts can have significantly different memory footprints based on design choices.  
    * For example, consider wide and deep CNNs. A wide model has a large number of channels in the convolutional layers, while a deep model has a large number of layers. Both models can have similar parameter counts, but the wide model will have a larger memory footprint due to the larger number of channels. $^1$

## Usage
1. You can directly see the `final_df.csv` file for the benchmarking results of the models on **`imagenet-real`** dataset.**Imagenet-real** is the usual ImageNet-1k validation set with a fresh new set of labels intended to improve on mistakes in the original annotation process.
      * Source: https://github.com/google-research/reassessed-imagenet
      * Paper: "Are we done with ImageNet?" - https://arxiv.org/abs/2006.07159

2. You can also run the `add_model_size.py` file to add the model size to any of the csvs you obtain or create from the benchmarks. The script uses `beautifulsoup4` to scrape the model size from the HuggingFace model hub.
   1. You can install the dependencies using the following command,
      ```bash
      pip install -r requirements.txt
      ```
    2. You can run the script using the following command,
      ```bash
        python add_model_size.py --orig_score_path <path_to_original_csv>  --hf_access_token <huggingface_access_token> --temp_file_path <path_to_store_temp_files>  --output_path <path_to_store_final_csv>
      ```
    * The `--hf_access_token` is optional. You can get it from your HuggingFace account. It is used to increase the rate limit of the HuggingFace API.
    * The `--temp_file_path` is optional. It is used to store the temporary files that are used to scrape the model size from the HuggingFace model hub. If not provided, the temporary files are stored in the current directory. These will be deleted after the entire process is done.
    * The `--output_path` is used to store the final csv file. If not provided, the final csv file is stored in the current directory.
    * The `--orig_score_path` is the path to the original csv file that you want to add the model size to.



$1.$ Though the statement is generally true in practice, it is not always the case. The wider models can indeed be designed to have a smaller memory footprint than the deeper model. This was just an example. 