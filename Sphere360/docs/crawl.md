# Data Crawling


![DataCrawl](img/crawl.png)


- [Data Crawling](#data-crawling)
  - [Search](#search)
  - [Channel-Based Crawling](#channel-based-crawling)
    - [Analyze channels based on search results](#analyze-channels-based-on-search-results)
    - [Get Video IDs for channels](#get-video-ids-for-channels)
    - [Get test video metadata](#get-test-video-metadata)
    - [Download test videos \& Check](#download-test-videos--check)
    - [Get Video IDs based on Channel IDs](#get-video-ids-based-on-channel-ids)
    - [Download](#download)
  - [Video-Based Crawling](#video-based-crawling)
    - [Filter by Blacklist](#filter-by-blacklist)
    - [Download Test Videos \& Verification \& Full Download](#download-test-videos--verification--full-download)
  - [Download](#download-1)
    - [Full Video Download](#full-video-download)
    - [Batch Download Function](#batch-download-function)


## Search

**Path:** [toolset/crawl/search/search.sh](../toolset/crawl/search/search.sh)

**Description:** This script uses the YouTube API to search for 360-degree videos based on a  predefined list of keywords and suffix terms. Note that the retrieved  videos are confirmed as 360-degree videos but **do not guarantee support for Spatial Audio**. To filter videos that support Spatial Audio, use a downloader (e.g., `yt-dlp` with an audio channel count filter).

**Usage Instructions:**

1. Prepare a keyword list file (one keyword per line).
2. Modify the parameters in `search.sh`, such as `keyword_file`, `postfix`(i.e. the qualifying term), and `output_dir`. Refer to the code comments for detailed parameter descriptions.
3. Run `bash search.sh` to execute the search. Results are saved in CSV format under the `output_dir` directory, organized by keyword.

## Channel-Based Crawling

### Analyze channels based on search results

**Path:** [toolset/crawl/channel/channel_analyzer.py](../toolset/crawl/channel/channel_analyzer.py)

**Description:** Analyzes frequently appearing channels based on search results from the `search` module. Outputs in CSV format.

**Usage Instructions:**

```bash
python channel_analyzer.py -i [input-dir] -o [output-file]
```

### Get Video IDs for channels

**Path:** [toolset/crawl/channel/get_channel_vids.py](../toolset/crawl/channel/get_channel_vids.py)

**Description:** Retrieves all 360-degree video IDs from the most promising channels (those containing the most search results). The output directory contains CSV files named by Channel ID, each containing all 360-degree video IDs for the corresponding channel.

**Usage Instructions:**

```bash
python get_channel_vids.py -i [input-csv] -o [out-dir] -t [threshold]
```

Where `threshold` specifies the minimum number of times a channel must appear in search results to be retained.

### Get test video metadata

**Path:** [toolset/crawl/channel/get_test_list.py](../toolset/crawl/channel/get_test_list.py)

**Description:** For each channel, randomly samples several video segments from the Channel ID's video ID information, and outputs in CSV format recognizable by download scripts.

**Usage Instructions:**

```bash
python get_test_list.py -i [in-dir] -o [out-dir] [-n [sample-size]]
```

Here `in-dir` contains the CSV files named by Channel ID obtained from the previous module, and `out-dir` will contain CSV files named by each Channel ID, with each file containing up to 10 Video IDs for the corresponding channel.

### Download test videos & Check

Use your preferred downloader or scripts from the `Download` section to download test segments, then perform manual verification or use the cleaning pipeline from the `Data Cleaning` section to verify channel quality, filtering out usable and unusable Channel IDs (both can be included in the Channel Black List).

### Get Video IDs based on Channel IDs

**Path:** [toolset/crawl/channel/get_channels_vids.py](../toolset/crawl/channel/get_channels_vids.py)

**Description:** Retrieves video IDs to be downloaded based on a list of available Channel IDs. All resulting Video IDs are guaranteed to be 360-degree videos. Due to API limitations, Spatial Audio support is verified during the download phase rather than this stage.

**Usage Instructions:**

```bash
python get_channels_vids.py -i [input-csv] -o [output-csv]
```

### Download

Use your preferred downloader or scripts from the `Download` section for video downloading.

## Video-Based Crawling

### Filter by Blacklist

**Path:** [toolset/crawl/filter/filter_exist.py](../toolset/crawl/filter/filter_exist.py)

**Description:**  

Utilizes the established video and channel blacklists from Channel-Based Crawling to filter out confirmed unnecessary video entries from search results, generating a smaller-scale list for manual review.

**Usage Instructions:**  

1. Prepare the Video ID blacklist database and Channel ID blacklist database (including all manually verified usable/unusable video or channel IDs). Modify `video_db_path` and `channel_db_path` in the script.  
2. Update the search results path `folder_path` (output from the `Search` module) and output file path `output_csv` in the script.  
3. Run `python filter_exist.py` to execute the filtering.  

### Download Test Videos & Verification & Full Download

Use a downloader of choice to fetch video clips for screening. After verification, proceed with full downloads of usable videos.  

## Download  

### Full Video Download  

**Path:** [toolset/crawl/download/download_list.sh](../toolset/crawl/download/download_list.sh)

**Description:**  Downloads all videos based on Video IDs provided in a CSV file. Supports multi-process downloading. Results are logged in `success_list.txt` and `fail_list.txt` in the current directory.  

**Usage Instructions:**  

Modify relevant parameters in [download_list.sh](../toolset/crawl/download/download_list.sh), then execute:  

```bash
bash download_list.sh
```

### Batch Download Function  

The `download_list_360` function in [toolset/crawl/core/download/download_list.py](../toolset/crawl/core/download/download_list.py) enables batch downloading. Refer to the function documentation for usage. Common configurations include:  

- **Clip Downloading**: Use `specify_start` and `time_interval` parameters (see function docs for details).  
- **Multi-process Downloading**: Set the number of parallel downloads via the `jobs` parameter.  
- **Custom Cookies**: Specify cookies using the `cookie` parameter.  
- **Proxy Configuration**: Configure proxy servers via the `proxy` parameter.  