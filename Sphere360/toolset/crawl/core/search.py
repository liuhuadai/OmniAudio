import json
import os

def search_videos(
    youtube_client,
    query,
    num_pages=1,
    max_results_per_page=50,
    output_tmp=False,
    use_cache=False,
    tmp_path=None,
):
    """
    Search videos using YouTube API and return video IDs, titles and other metadata
    
    :param youtube_client: Initialized YouTube API client object
    :param query: Search query keywords
    :param num_pages: Number of result pages to fetch
    :param max_results_per_page: Maximum results per page
    :return: List of dictionaries containing video IDs, titles and other metadata
    """
    if output_tmp or use_cache:
        if tmp_path is None:
            tmp_path = "tmp"
        if output_tmp:
            os.makedirs(tmp_path, exist_ok=True)

    if tmp_path is not None:
        tmp_path = os.path.join(tmp_path, "search")
        os.makedirs(tmp_path, exist_ok=True)

    video_info_list = []  # Store search results
    next_page_token = None  # Initialize pageToken for next page

    for i in range(num_pages):
        # Try reading from cache
        if use_cache:
            try:
                with open(os.path.join(tmp_path, f"search_{i}.json"), "r", encoding="utf-8") as f:
                    search_response = json.load(f)
                print(f"Using cache {os.path.join(tmp_path, f'search_{i}.json')}")
            except FileNotFoundError:
                search_response = None

        if search_response is None:
            # Execute video search
            search_request = youtube_client.search().list(
                part="snippet",
                q=query,
                type="video",  # Only return videos
                maxResults=max_results_per_page,  # Results per page
                pageToken=next_page_token,  # pageToken from previous page (None for first request)
            )
            search_response = search_request.execute()

        # Parse response to get video IDs, titles and metadata
        for item in search_response["items"]:
            video_info = {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "channel_id": item["snippet"]["channelId"],
                "channel_title": item["snippet"]["channelTitle"],
                "description": item["snippet"]["description"],
                "publish_time": item["snippet"]["publishedAt"],
            }
            video_info_list.append(video_info)

        if output_tmp:
            with open(os.path.join(tmp_path, f"search_{i}.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(search_response, indent=2))

        # Get next page token
        next_page_token = search_response.get("nextPageToken")

        # Exit early if no more pages
        if not next_page_token:
            break

    video_ids = [video["video_id"] for video in video_info_list]
    return video_ids, video_info_list


def search_videos_360(
    youtube, query, num_pages=1, max_results_per_page=50, output_tmp=False, use_cache=False, tmp_path=None
):
    """
    Search for 360-degree videos using YouTube API
    
    :param youtube: Initialized YouTube API client object
    :param query: Search query keywords
    :param num_pages: Number of result pages to fetch
    :param max_results_per_page: Maximum results per page
    :return: List of 360-degree video IDs
    """
    from .filters import filter_360

    video_ids, video_info_list = search_videos(
        youtube, query, num_pages, max_results_per_page, output_tmp, use_cache, tmp_path
    )

    video_ids = filter_360(youtube, video_ids, False)

    # Filter info
    video_info_dict = {
        video["video_id"]: video for video in video_info_list
    }
    video_info_list = []
    for video_id in video_ids:
        video_info_list.append(video_info_dict[video_id])

    return video_ids, video_info_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        help="Search query",
        default="Spatial Audio 360",
    )
    parser.add_argument(
        "--num-pages",
        type=int,
        help="Number of pages to search",
        default=2,
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Max results per page",
        default=50,
    )
    args = parser.parse_args()
    query = args.key
    num_pages = args.num_pages
    max_results_per_page = args.max_results
    from build import build_youtube

    youtube = build_youtube()

    video_ids, video_info_list = search_videos_360(
        youtube, query, num_pages, max_results_per_page, True
    )

    # Output results
    for video_id in video_ids:
        print(video_id)
    for video_info in video_info_list:
        print(str(video_info))