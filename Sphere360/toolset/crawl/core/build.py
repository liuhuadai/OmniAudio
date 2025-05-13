from googleapiclient.discovery import build
import os
import requests
from googleapiclient.http import HttpRequest
import httplib2

__API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"  # Enter your YouTube API key here

def build_youtube(api_key=None, proxy_host=None, proxy_port=None):
    """Initialize and configure the YouTube API client
    
    Args:
        api_key (str, optional): YouTube Data API key. Uses default if not provided.
        proxy_host (str, optional): Proxy server host address
        proxy_port (int, optional): Proxy server port number
        
    Returns:
        googleapiclient.discovery.Resource: Configured YouTube API client instance
    """
    if api_key is None:
        api_key = __API_KEY
        
    # Configure proxy settings
    http = httplib2.Http(
        proxy_info=httplib2.ProxyInfo(
            proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
        )
    )

    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    # Initialize YouTube API client
    youtube = build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        developerKey=api_key,
        http=http,
    )

    return youtube