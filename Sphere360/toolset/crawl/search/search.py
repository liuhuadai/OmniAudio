from tqdm import tqdm
import json
import os
import csv
import sys
sys.path.append('..')
from core import search, build

def output_to_csv(video_info_list, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        if len(video_info_list) == 0:
            return
        dict_writer = csv.DictWriter(f, fieldnames=video_info_list[0].keys())
        dict_writer.writeheader()
        for video_info in video_info_list:
            dict_writer.writerow(video_info)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file with keywords')
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-n', '--num-pages', type=int, default=1)
    parser.add_argument('-p', '--postfix', type=str, default=None)
    parser.add_argument('-t', '--tmp-path', type=str)
    args = parser.parse_args()
    output = args.output
    num_pages = args.num_pages
    tmp_path = args.tmp_path

    with open(args.input, 'r') as f:
        keywords = f.readlines()
    keywords = [keyword.strip() for keyword in keywords]

    print(f'Searching for {len(keywords)} keywords')

    youtube = build.build_youtube()

    pbar = tqdm(keywords)
    for keyword in pbar:
        if args.postfix:
            keyword += args.postfix
        pbar.set_description(f'{keyword}') 
        
        out_path = os.path.join(output, f'{keyword}.csv')
        if os.path.exists(out_path):
            print(f"Skip {keyword}")
            continue

        video_ids, video_info_list = search.search_videos_360(
            youtube, keyword, num_pages, 50, True, True, os.path.join(tmp_path, f'{keyword}')
        )
        out_path = os.path.join(output, f'{keyword}.csv')
        output_to_csv(video_info_list, out_path)
    
    print(f'Finished searching for {len(keywords)} keywords')
