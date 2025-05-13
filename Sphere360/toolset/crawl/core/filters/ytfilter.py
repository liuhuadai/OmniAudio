import json

def filter_360(youtube, video_ids, output_tmp=False):
    """
    Filters and returns only 360-degree video IDs from the input list
    
    :param youtube: Initialized YouTube API client object
    :param video_ids: List of video IDs to filter
    :param output_tmp: If True, saves intermediate results to temporary files
    :return: List containing only 360-degree video IDs
    """
    vr_video_ids = []  # List to store 360-degree video IDs

    # Process videos in batches of 50 (YouTube API limit)
    for i in range(0, len(video_ids), 50):
        batch_video_ids = video_ids[i:i + 50]  # Get current batch of 50 video IDs

        # Request video details in batch
        request = youtube.videos().list(
            part="contentDetails",
            id=",".join(batch_video_ids),  # Comma-separated video IDs
        )
        response = request.execute()

        # Optionally save intermediate results
        if output_tmp:
            with open(f"tmp/get_video_detail_{i // 50}.json", "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)

        # Check projection type for each video
        for item in response.get("items", []):
            if item["contentDetails"].get("projection", "") == "360":
                vr_video_ids.append(item["id"])

    return vr_video_ids


# Test function
if __name__ == "__main__":
    from build import build_youtube

    # Initialize YouTube API client
    youtube = build_youtube()

    # Test video IDs (mix of regular and 360 videos)
    test_video_ids = [ # example videos
        "spYqJw3WpCI",
        "8AEhFvFMwBo",
        "XToK00VcBI8",
    ]
    
    # Filter 360 videos
    vr_videos = filter_360(youtube, test_video_ids, output_tmp=True)

    # Print results
    print("\n360-degree Video IDs:")
    for vid in vr_videos:
        print(f"- {vid}")