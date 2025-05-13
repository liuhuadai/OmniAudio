import os
import json


def get_channel_video_ids(youtube, channel_id, output_tmp=False):
    """
    Retrieve all video IDs from a specified YouTube channel.

    Args:
        youtube: Initialized YouTube API client
        channel_id: YouTube channel ID
        output_tmp: Whether to save temporary JSON outputs
    Returns:
        List of video IDs
    """
    video_ids = []

    # Get channel's upload playlist ID
    request = youtube.channels().list(part="contentDetails", id=channel_id)
    response = request.execute()

    if output_tmp:
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/get_playlist.json", "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)

    upload_playlist_id = response["items"][0]["contentDetails"][
        "relatedPlaylists"
    ]["uploads"]

    # Get all videos from upload playlist
    next_page_token = None
    while True:
        playlist_request = youtube.playlistItems().list(
            part="snippet",
            playlistId=upload_playlist_id,
            maxResults=50,  # Max 50 videos per request
            pageToken=next_page_token,
        )

        playlist_response = playlist_request.execute()

        if output_tmp:
            token_str = next_page_token if next_page_token else "first_page"
            with open(
                f"tmp/get_video_next_{token_str}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(playlist_response, f, indent=2)

        # Extract video IDs
        for item in playlist_response["items"]:
            video_ids.append(item["snippet"]["resourceId"]["videoId"])

        # Check for more pages
        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids


def get_channel_video_ids_360(youtube, channel_id, output_tmp=False):
    """Get only 360-degree video IDs from a channel"""
    from . import filters

    video_ids = get_channel_video_ids(youtube, channel_id, output_tmp)
    return filters.filter_360(youtube, video_ids, output_tmp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrieve video IDs from YouTube channel"
    )
    parser.add_argument(
        "--channel-id",
        type=str,
        required=True,
        help="YouTube channel ID to process"
    )
    args = parser.parse_args()

    import build
    youtube = build.build_youtube()

    # Test functionality
    video_ids = get_channel_video_ids(youtube, args.channel_id, True)
    print(video_ids)