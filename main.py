import ssl
import time
import requests
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from bs4 import BeautifulSoup as soup


ssl._create_default_https_context = ssl._create_unverified_context


def get_model_size(model_name, hf_access_token):
    url = f"https://huggingface.co/timm/{model_name}/tree/main"
    headers = {"Authorization": f"Bearer {hf_access_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        page_soup = soup(response.content, "html.parser")
        all_file_sizes = page_soup.find_all('a', {'title': 'Download file'})
        selected_file = [file for file in all_file_sizes if "pytorch_model.bin" in file['href']]
        model_size = selected_file[0].text
        model_size = model_size.split("\n")[0]
    elif response.status_code == 404:
        model_size = "Model Not Found"
    elif response.status_code == 401:
        model_size = "Model Authorization Error"
    else:
        raise Exception(f"Error in getting model size for {model_name}")
    return model_size


def get_model_sizes(model_names, size_csv_path, hf_access_token):
    model_sizes = []
    count = 0
    # get the idx from whcih we have to resume from the csv file:
    try:
        model_size_df = pd.read_csv(size_csv_path, header=None)
        model_shape = model_size_df.shape
        model_resume_idx = model_shape[0]
        print(f"Resuming from index {model_resume_idx}")
    except FileNotFoundError:
        model_resume_idx = 0

    if model_resume_idx == len(model_names):
        print("All models have been scraped")
        return model_size_df
    else:
        for model_name in tqdm(model_names[model_resume_idx:]):
            model_size = get_model_size(model_name, hf_access_token)
            model_sizes.append(model_size)
            with open(size_csv_path, "a") as f:
                f.write(f"{model_name},{model_size}\n")
            count += 1
            if count % 45 == 0:
                time.sleep(30)

        model_size_df = pd.read_csv(size_csv_path)
        return model_size_df


def get_size_in_mb(combined_df):
    model_sizes = combined_df.model_size[combined_df.model_size.notnull()].copy()

    conversion_df = pd.DataFrame(columns=["original_size"])
    conversion_df['original_size'] = model_sizes
    conversion_df.reset_index(inplace=True, drop=True)
    conversion_df['metric'] = conversion_df.original_size.apply(
        lambda x: x.split(" ")[1] if x is not np.nan else np.nan)
    conversion_df['size'] = conversion_df.original_size.apply(
        lambda x: float(x.split(" ")[0]) if x is not np.nan else np.nan)
    conversion_df['multiplier'] = conversion_df.metric.apply(lambda x: 1 if x == "MB" else 1024)
    conversion_df["size_in_mb"] = conversion_df['size'] * conversion_df['multiplier']

    return conversion_df['size_in_mb'].values


def get_combined_df(score_df, model_size_df):
    combined_df = pd.concat([score_df, model_size_df], axis=1)
    combined_df.rename(columns={1: "model_size"}, inplace=True)
    combined_df.drop(columns=[0], inplace=True)
    combined_df = combined_df.loc[:, ['model', 'model_size', 'top1', 'top1_err', 'top5', 'top5_err', 'param_count',
                                      'img_size', 'crop_pct', 'interpolation', 'top1_diff', 'top5_diff',
                                      'rank_diff']]

    combined_df.model_size = combined_df.model_size.apply(
        lambda x: np.nan if x in ["Model Not Found", "Model Authorization Error"] else x)
    return combined_df


def get_final_df(score_df_path, scraped_size_path, hf_access_token):
    score_df = pd.read_csv(score_df_path)

    # get Model Sizes from HunggingFace library
    model_size_df = get_model_sizes(score_df['model'].values,
                                    scraped_size_path,
                                    hf_access_token)

    # combine the original and final_dfs
    combined_df = get_combined_df(score_df, model_size_df)

    # convert the model sizes to MB. Leaving out the models which are not found.
    sizes_in_mb = get_size_in_mb(combined_df)
    nonnull_idx = combined_df.model_size[combined_df.model_size.notnull()].index
    combined_df.loc[nonnull_idx, 'model_size_in_mb'] = sizes_in_mb

    # A final dataframe with only the required columns
    final_cols = ['model', 'model_size_in_mb', 'top1', 'top1_err', 'top5', 'top5_err', 'param_count',
                  'img_size', 'crop_pct', 'interpolation', 'top1_diff', 'top5_diff',
                  ]

    final_df = combined_df.loc[:, final_cols].copy()
    return final_df

def parse_args(parser):
    parser.add_argument("--orig_score_path", type=str, default="results-imagenet-real.csv",
                        help="Path to the original score csv file")
    parser.add_argument("--scraped_size_path", type=str, default="model_sizes.csv",
                        help="Path to the scraped model sizes csv file")
    parser.add_argument("--hf_access_token", type=str, default='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',
                        help="Hugging Face access token")
    parser.add_argument("--output_path", type=str, default="final_df.csv",
                        help="Path to save the final dataframe")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    output_path = args.output_path
    HF_ACCESS_TOKEN = args.hf_access_token
    final_df = get_final_df(args.orig_score_path, args.scraped_size_path, HF_ACCESS_TOKEN)
    final_df.to_csv(output_path, index=False)