import csv
import os
from tqdm import tqdm
import subprocess
import sys
from multiprocessing import Pool, Lock
from typing import List, Dict, Any, Optional


class DownloadError(Exception):
    def __init__(self, args: Dict[str, Any]):
        self.items = args["items"]
        self.result = args["result"]


def _check_size(check_files: List[str], size=1024, remove: bool = True):
    """
    check if all files is larger than the limit,
    :param remove: if True, remove files that less than that size.
    """

    check_success = True
    for file_name in check_files:
        if not os.path.exists(file_name):
            check_success = False
            continue
        if os.path.getsize(file_name) < size:
            check_success = False
            if remove:
                os.remove(file_name)
            else:
                break
    return check_success

def download_video_process(args: Dict):
    return download_video(**args)


def download_video_segments_process(args: Dict):
    return download_video_segments(**args)


def download_4ch_segments_process(args: Dict):
    return download_4ch_segments(**args)

def download_360_segments_process(args: Dict):
    return download_360_segments(**args)

def download_360_process(args: Dict):
    return download_360(**args)

def download_video(
    video_id: str,
    output_folder: str,
    ext: str = None,
    proxy: str = None,
    try_time: int = 2,
    format_code: str = "bv+ba",
    check_size: bool = True,
    skip_exists: bool = True,
    time_out: int = 30,
) -> List[str]:
    """
    :param skip_exists: whether to skip the existing files. If True, ext should be specified.
    :param ext: extension of the output file without dot.
    :return: List of success file_ids
    """
    # Check format code with ext
    if ext is not None:
        if format_code.find("[ext=") != -1:
            if format_code.find(f"[ext={ext}]") == -1:
                raise ValueError(
                    f"Format code {format_code} does not match the extension {ext}"
                )
        else:
            format_code += f"[ext={ext}]"

    file_items = [video_id]
    file_path_base = os.path.join(output_folder, video_id)
    if skip_exists:
        if ext is None:
            raise ValueError("ext should be specified when skip_exists is True")
        if os.path.exists(file_path_base + "." + ext):
            print(f"{video_id} already exists")
            return file_items

    # specify command
    cmd = [
        "yt-dlp",
        "-f",
        format_code,
        f"https://www.youtube.com/watch?v={video_id}",
        "-o",
        f"{file_path_base}.%(ext)s",
        "--socket-timeout",
        str(time_out),
        "--abort-on-error",
        "--abort-on-unavailable-fragments",
        "-N",
        "4",
    ]

    if proxy is not None:
        cmd += ["--proxy", proxy]

    try_id = 0
    success = False
    result = None
    while try_id < try_time and not success:
        print(f"Downloading video {video_id}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        # result = subprocess.run(cmd)
        if result.returncode == 0:
            if check_size:
                file_names = [
                    os.path.join(output_folder, file_item + "." + ext)
                    for file_item in file_items
                ]
                success = _check_size(file_names, remove=True)
            else:
                success = True
        if not success:
            print(
                f"[WARNING] Failed to download video items {video_id}, retrying...({try_id + 1}/{try_time})"
            )
        try_id += 1

    if not success:
        raise DownloadError(
            {
                "items": file_items,
                "result": result,
            }
        )
    return file_items


def download_4ch_segments(
    video_id: str,
    output_folder: str,
    start_times: List[int],
    time_interval: int = 10,
    proxy: str = None,
    try_time: int = 4,
    check_size: bool = True,
    skip_exists: bool = True,
    time_out: int = 30,
):
    """
    :param skip_exists: whether to skip the existing files. If True, ext should be specified.
    :param check_size: whether to check file size. If True, remove files that less than 1024 bytes and ext should be specified.
    :param ext: extension of the output file without dot.
    :return: List of success file_ids
    """
    format_code = "ba*[audio_channels=4]"
    ext = "webm"

    origin_file_items = [
        video_id + "_" + str(start_time) for start_time in start_times
    ]
    file_path_base = os.path.join(output_folder, video_id)
    if skip_exists:
        original_start_times = start_times
        start_times = []
        for st in original_start_times:
            if os.path.exists(file_path_base + "_" + str(st) + "." + ext):
                print(f"{video_id}_{st} already exists")
            else:
                start_times.append(st)

    if len(start_times) == 0:
        return origin_file_items

    end_times = [t + time_interval for t in start_times]

    file_items = [
        video_id + "_" + str(start_time) for start_time in start_times
    ]

    # specify command
    cmd = [
        "yt-dlp",
        "-f",
        format_code,
        f"https://www.youtube.com/watch?v={video_id}",
        "-o",
        f"{file_path_base}_%(section_start)d.%(ext)s",
        "--socket-timeout",
        str(time_out),
        "--abort-on-error",
        "--abort-on-unavailable-fragments",
        "-N",
        "4",
        "--force-keyframes-at-cuts",
        "--extractor-args",
        "youtube:player_client=all",
        "--merge-output-format",
        "webm",
    ]

    if proxy is not None:
        cmd += ["--proxy", proxy]

    if start_times is not None:
        for start_time, end_time in zip(start_times, end_times):
            cmd += [
                "--download-sections",
                f"*{start_time}-{end_time}",
            ]

    try_id = 0
    success = False
    result = None
    while try_id < try_time and not success:
        print(f"Downloading video items {video_id}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if check_size:
                file_names = [
                    os.path.join(output_folder, file_item + "." + ext)
                    for file_item in file_items
                ]
                success = _check_size(file_names, remove=True)
            else:
                success = True
        if not success:
            print(
                f"[WARNING] Failed to download video items {video_id}, retrying...({try_id + 1}/{try_time})"
            )
        try_id += 1

    if not success:
        raise DownloadError(
            {
                "items": file_items,
                "result": result,
            }
        )
    return file_items

def download_360(
    video_id: str,
    output_folder: str,
    proxy: str = None,
    try_time: int = 2,
    check_size: bool = True,
    skip_exists: bool = True,
    time_out: int = 30,
    cookie=None
) -> List[str]:
    """
    :param skip_exists: whether to skip the existing files. If True, ext should be specified.
    :param ext: extension of the output file without dot.
    :return: List of success file_ids
    """
    # Check format code with ext
    format_code = "bv[height<=1080]+ba[audio_channels=4]"
    ext = "webm"

    file_items = [video_id]
    file_path_base = os.path.join(output_folder, video_id)
    if skip_exists:
        if ext is None:
            raise ValueError("ext should be specified when skip_exists is True")
        if os.path.exists(file_path_base + "." + ext):
            print(f"{video_id} already exists")
            return file_items

    print(file_path_base)
    # specify command
    cmd = [
        "yt-dlp",
        "-f",
        format_code,
        f"https://www.youtube.com/watch?v={video_id}",
        "-o",
        f"{file_path_base}.%(ext)s",
        "--socket-timeout",
        str(time_out),
        "--abort-on-error",
        "--abort-on-unavailable-fragments",
        "-N",
        "4",
        "--extractor-args",
        "youtube:player_client=all",
        "--merge-output-format",
        ext,
    ]

    if proxy is not None:
        cmd += ["--proxy", proxy]
    
    if cookie is not None:
        cmd += ["--cookies", cookie]

    try_id = 0
    success = False
    result = None
    while try_id < try_time and not success:
        print(f"Downloading video {video_id}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        # result = subprocess.run(cmd)
        if result.returncode == 0:
            if check_size:
                file_names = [
                    os.path.join(output_folder, file_item + "." + ext)
                    for file_item in file_items
                ]
                success = _check_size(file_names, remove=True)
            else:
                success = True
        if not success:
            print(
                f"[WARNING] Failed to download video items {video_id}, retrying...({try_id + 1}/{try_time})"
            )
        try_id += 1

    if not success:
        raise DownloadError(
            {
                "items": file_items,
                "result": result,
            }
        )
    return file_items


def download_360_segments(
    video_id: str,
    output_folder: str,
    start_times: List[int],
    time_interval: int = 10,
    proxy: str = None,
    try_time: int = 2,
    check_size: bool = True,
    skip_exists: bool = True,
    time_out: int = 30,
    cookie=None,
):
    """
    :param skip_exists: whether to skip the existing files. If True, ext should be specified.
    :param check_size: whether to check file size. If True, remove files that less than 1024 bytes and ext should be specified.
    :param ext: extension of the output file without dot.
    :return: List of success file_ids
    """
    format_code = "bv[height<=1440]+ba[audio_channels=4]"
    ext = "webm"

    origin_file_items = [
        video_id + "_" + str(start_time) for start_time in start_times
    ]
    file_path_base = os.path.join(output_folder, video_id)
    if skip_exists:
        original_start_times = start_times
        start_times = []
        for st in original_start_times:
            if os.path.exists(file_path_base + "_" + str(st) + "." + ext):
                print(f"{video_id}_{st} already exists")
            else:
                start_times.append(st)

    if len(start_times) == 0:
        return origin_file_items

    end_times = [t + time_interval for t in start_times]

    file_items = [
        video_id + "_" + str(start_time) for start_time in start_times
    ]

    # specify command
    cmd = [
        "yt-dlp",
        "-f",
        format_code,
        f"https://www.youtube.com/watch?v={video_id}",
        "-o",
        f"{file_path_base}_%(section_start)d.%(ext)s",
        "--socket-timeout",
        str(time_out),
        "--abort-on-error",
        "--abort-on-unavailable-fragments",
        "-N",
        "4",
        "--force-keyframes-at-cuts",
        "--extractor-args",
        "youtube:player_client=all",
        "--merge-output-format",
        "webm",
    ]

    if proxy is not None:
        cmd += ["--proxy", proxy]
    
    if cookie is not None:
        cmd += ["--cookies", cookie]

    if start_times is not None:
        for start_time, end_time in zip(start_times, end_times):
            cmd += [
                "--download-sections",
                f"*{start_time}-{end_time}",
            ]

    try_id = 0
    success = False
    result = None
    while try_id < try_time and not success:
        print(f"Downloading video items {video_id}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if check_size:
                file_names = [
                    os.path.join(output_folder, file_item + "." + ext)
                    for file_item in file_items
                ]
                success = _check_size(file_names, remove=True)
            else:
                success = True
        if not success:
            stderr = result.stderr
            if stderr.find("format is not available") != -1:
                print(
                    f"[WARNING] Skip {video_id} since format not available({try_id + 1}/{try_time})"
                )
                break
            print(
                f"[WARNING] Failed to download video items {video_id}, retrying...({try_id + 1}/{try_time})"
            )
        try_id += 1

    if not success:
        raise DownloadError(
            {
                "items": file_items,
                "result": result,
            }
        )
    return file_items


def download_video_segments(
    video_id: str,
    output_folder: str,
    start_times: List[int],
    ext: str = None,
    time_interval: int = 10,
    proxy: str = None,
    try_time: int = 2,
    check_size: bool = True,
    skip_exists: bool = True,
    time_out: int = 30,
    format_code: str = "bv+ba",
) -> List[str]:
    """
    :param skip_exists: whether to skip the existing files. If True, ext should be specified.
    :param check_size: whether to check file size. If True, remove files that less than 1024 bytes and ext should be specified.
    :param ext: extension of the output file without dot.
    :return: List of success file_ids
    """
    # Check check_size with ext
    if check_size and ext is None:
        raise ValueError("ext should be specified when check_size is True")

    # Check format code with ext
    if ext is not None:
        if format_code.find("[ext=") != -1:
            if format_code.find(f"[ext={ext}]") == -1:
                raise ValueError(
                    f"Format code {format_code} does not match the extension {ext}"
                )
        else:
            format_code += f"[ext={ext}]"

    origin_file_items = [
        video_id + "_" + str(start_time) for start_time in start_times
    ]
    file_path_base = os.path.join(output_folder, video_id)
    if skip_exists:
        if ext is None:
            raise ValueError("ext should be specified when skip_exists is True")
        original_start_times = start_times
        start_times = []
        for st in original_start_times:
            if os.path.exists(file_path_base + "_" + str(st) + "." + ext):
                print(f"{video_id}_{st} already exists")
            else:
                start_times.append(st)

    if len(start_times) == 0:
        return origin_file_items

    end_times = [t + time_interval for t in start_times]

    file_items = [
        video_id + "_" + str(start_time) for start_time in start_times
    ]

    # Specify command
    cmd = [
        "yt-dlp",
        "-f",
        format_code,
        f"https://www.youtube.com/watch?v={video_id}",
        "-o",
        f"{file_path_base}_%(section_start)d.%(ext)s",
        "--socket-timeout",
        str(time_out),
        "--abort-on-error",
        "--abort-on-unavailable-fragments",
        "-N",
        "4",
        "--force-keyframes-at-cuts",
    ]

    if proxy is not None:
        cmd += ["--proxy", proxy]

    if start_times is not None:
        for start_time, end_time in zip(start_times, end_times):
            cmd += [
                "--download-sections",
                f"*{start_time}-{end_time}",
            ]

    try_id = 0
    success = False
    result = None
    while try_id < try_time and not success:
        print(f"Downloading video items {video_id}...")
        result = subprocess.run(
            cmd,
            # capture_output=True,
            # text=True,
        )
        if result.returncode == 0:
            if check_size:
                file_names = [
                    os.path.join(output_folder, file_item + "." + ext)
                    for file_item in file_items
                ]
                success = _check_size(file_names, remove=True)
            else:
                success = True
        if not success:
            print(
                f"[WARNING] Failed to download video items {video_id}, retrying...({try_id + 1}/{try_time})"
            )
        try_id += 1

    if not success:
        raise DownloadError(
            {
                "items": file_items,
                "result": result,
            }
        )
    return file_items


def get_video_ids(input_file: str):
    video_ids = []
    if input_file.endswith(".csv"):  # CSV file
        with open(input_file, "r") as file:
            csvreader = csv.DictReader(file)
            for row in csvreader:
                video_ids.append(row["video_id"])
    else:  # Default format
        with open(input_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                video_ids.append(line.strip())

    return video_ids


def get_video_ids_and_start_times(input_file: str):
    video_ids = []
    start_times = []
    lines = []
    if input_file.endswith(".csv"):  # CSV file
        with open(input_file, "r") as file:
            csvreader = csv.DictReader(file)
            for row in csvreader:
                lines.append(row["file_id"])
    else:  # Default format
        with open(input_file, "r") as file:
            lines = file.readlines()

    for line in lines:
        parts = line.rsplit("_", 1)
        if len(parts) != 2:
            continue
        video_ids.append(parts[0])
        start_times.append(int(parts[1]))

    return video_ids, start_times


def get_video_ids_and_start_times_list(input_file: str):
    file_ids = []
    start_times_list = []
    with open(input_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        file_ids.append(parts[0])
        start_times = parts[1].split(",")
        start_times_list.append([int(t) for t in start_times if t.isdigit()])
    return file_ids, start_times_list


def download_list_4ch(
    input_file,
    output_folder,
    start_index=None,
    end_index=None,
    specify_start: str = None,
    proxy=None,
    time_interval=None,
    fail_list_name="fail_list.txt",
    success_list_name="success_list.txt",
    jobs=1,
):
    """
    Download video items from a list of video ids. Can specify the start time of each video.

    :param input_file: input file path. Can be a csv file with column 'video_id'/'file_id'(start time specified)
        or a txt file with each line as a video id. See specify_start for more details about format.
    :param output_folder: output folder path.
    :param start_index: start index of the video list, None means the start of the list. The list will be slice by [start_index:end_index)
    :param end_index: end index of the video list, None means the end of the list. The list will be slice by [start_index:end_index)
    :param specify_start: If specified, time_interval should be specified as well.
        None: (Default) not to specify the start time. Each item is a video_id.
        "single": specify single start time for each video_id.
            For non-csv file, format each line by f'{video_id}_{start_time}'.
            For CSV file, title by 'file_id' and format the same.
        "multiple": specify multiple start times for each video_id. Only non-csv file is supported in this mode.
            Format each line as f'{video_id} {start_times}', where start_times is a list of start times separated by ','.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"speicify_start: {specify_start}")
    if specify_start is not None:
        if time_interval is None:
            raise ValueError(
                "time_interval should be specified when specify_start is not None."
            )
        if specify_start == "single":
            video_ids, start_times = get_video_ids_and_start_times(input_file)
            start_times_list = [[start_time] for start_time in start_times]
        elif specify_start == "multiple":
            video_ids, start_times_list = get_video_ids_and_start_times_list(
                input_file
            )
        else:
            raise ValueError("Invalid specify_start value.")
    else:
        video_ids = get_video_ids(input_file)
        start_times_list = None

    start_index = 0 if start_index is None else start_index
    end_index = len(video_ids) if end_index is None else end_index
    if start_index > 0:
        print(f"Start from index:{start_index}")
    if end_index != len(video_ids):
        print(f"End to index:{end_index}")
    video_ids = video_ids[start_index:end_index]
    if specify_start:
        start_times_list = start_times_list[start_index:end_index]

    print(f"Downloading {len(video_ids)} videos into {output_folder}")

    # create success & fail list
    print(f"Success files written into {success_list_name}")
    print(f"Fail files written into {fail_list_name}")
    with open(success_list_name, "w") as f:
        pass
    with open(fail_list_name, "w") as f:
        pass

    pbar = tqdm(total=len(video_ids))
    success_list = []
    fail_list = []
    (
        pbar_lock,
        success_lock,
        fail_lock,
    ) = (
        Lock(),
        Lock(),
        Lock(),
    )

    def download_success(file_ids):
        nonlocal pbar, success_list_name, success_list, success_list, pbar_lock
        with pbar_lock:
            pbar.update(1)
        with success_lock:
            for file_id in file_ids:
                success_list.append(file_id)
                with open(success_list_name, "a") as f:
                    f.write(f"{file_id}\n")

    def download_fail(
        error: DownloadError,
    ):
        nonlocal pbar, fail_list_name, fail_list, fail_lock, pbar_lock
        with pbar_lock:
            pbar.update(1)
        if not isinstance(error, DownloadError):
            print(f"[Error]: {error}")
            raise error
        file_ids = error.items
        with fail_lock:
            for file_id in file_ids:
                fail_list.append(file_id)
                with open(fail_list_name, "a") as f:
                    f.write(f"{file_id}\n")

    # start downloading
    if specify_start is None:

        def arg_gen(video_ids):
            for video_id in video_ids:
                yield {
                    "video_id": video_id,
                    "output_folder": output_folder,
                    "proxy": proxy,
                }

        arg_iter = arg_gen(video_ids)
    else:

        def arg_gen(video_ids, start_times_list):
            for video_id, start_times in zip(video_ids, start_times_list):
                yield {
                    "video_id": video_id,
                    "start_times": start_times,
                    "output_folder": output_folder,
                    "proxy": proxy,
                    "time_interval": time_interval,
                }

        arg_iter = arg_gen(video_ids, start_times_list)

    # for arg in arg_iter:
    #     try:
    #         result = (download_video_process if specify_start is None else download_video_segments_process)(arg)
    #         download_success(result)
    #     except DownloadError as e:
    #         download_fail(e)

    p = Pool(jobs)
    for arg in arg_iter:
        p.apply_async(
            download_4ch_segments_process,
            args=(arg,),
            callback=download_success,
            error_callback=download_fail,
        )
    p.close()
    p.join()

    print("Finished downloading.")

    # output fail status
    success_list.sort()
    with open(success_list_name, "w") as f:
        for item in success_list:
            f.write(f"{item}\n")
    print(
        f"{len(success_list)} success files written into {success_list_name}."
    )
    fail_list.sort()
    with open(fail_list_name, "w") as f:
        for item in fail_list:
            f.write(f"{item}\n")
    print(f"{len(fail_list)} fail files written into {fail_list_name}.")

def download_list_360(
    input_file,
    output_folder,
    start_index=None,
    end_index=None,
    specify_start: str = None,
    proxy=None,
    time_interval=None,
    fail_list_name="fail_list.txt",
    success_list_name="success_list.txt",
    jobs=1,
    cookie=None,
):
    """
    Download video items from a list of video ids. Can specify the start time of each video.

    :param input_file: input file path. Can be a csv file with column 'video_id'/'file_id'(start time specified)
        or a txt file with each line as a video id. See specify_start for more details about format.
    :param output_folder: output folder path.
    :param start_index: start index of the video list, None means the start of the list. The list will be slice by [start_index:end_index)
    :param end_index: end index of the video list, None means the end of the list. The list will be slice by [start_index:end_index)
    :param specify_start: If specified, time_interval should be specified as well.
        None: (Default) not to specify the start time. Each item is a video_id.
        "single": specify single start time for each video_id.
            For non-csv file, format each line by f'{video_id}_{start_time}'.
            For CSV file, title by 'file_id' and format the same.
        "multiple": specify multiple start times for each video_id. Only non-csv file is supported in this mode.
            Format each line as f'{video_id} {start_times}', where start_times is a list of start times separated by ','.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"speicify_start: {specify_start}")
    if specify_start is not None:
        if time_interval is None:
            raise ValueError(
                "time_interval should be specified when specify_start is not None."
            )
        if specify_start == "single":
            video_ids, start_times = get_video_ids_and_start_times(input_file)
            start_times_list = [[start_time] for start_time in start_times]
        elif specify_start == "multiple":
            video_ids, start_times_list = get_video_ids_and_start_times_list(
                input_file
            )
        else:
            raise ValueError("Invalid specify_start value.")
    else:
        video_ids = get_video_ids(input_file)
        start_times_list = None

    start_index = 0 if start_index is None else start_index
    end_index = len(video_ids) if end_index is None else end_index
    if start_index > 0:
        print(f"Start from index:{start_index}")
    if end_index != len(video_ids):
        print(f"End to index:{end_index}")
    video_ids = video_ids[start_index:end_index]
    if specify_start:
        start_times_list = start_times_list[start_index:end_index]

    print(f"Downloading {len(video_ids)} videos into {output_folder}")

    # create success & fail list
    print(f"Success files written into {success_list_name}")
    print(f"Fail files written into {fail_list_name}")
    with open(success_list_name, "w") as f:
        pass
    with open(fail_list_name, "w") as f:
        pass

    pbar = tqdm(total=len(video_ids))
    success_list = []
    fail_list = []
    (
        pbar_lock,
        success_lock,
        fail_lock,
    ) = (
        Lock(),
        Lock(),
        Lock(),
    )

    def download_success(file_ids):
        nonlocal pbar, success_list_name, success_list, success_list, pbar_lock
        with pbar_lock:
            pbar.update(1)
        with success_lock:
            for file_id in file_ids:
                success_list.append(file_id)
                with open(success_list_name, "a") as f:
                    f.write(f"{file_id}\n")

    def download_fail(
        error: DownloadError,
    ):
        nonlocal pbar, fail_list_name, fail_list, fail_lock, pbar_lock
        with pbar_lock:
            pbar.update(1)
        if not isinstance(error, DownloadError):
            print(f"[Error]: {error}")
            raise error
        file_ids = error.items
        with fail_lock:
            for file_id in file_ids:
                fail_list.append(file_id)
                with open(fail_list_name, "a") as f:
                    f.write(f"{file_id}\n")
                print(f"Fail to download {file_id}: {error.result.stderr}")

    # start downloading
    if specify_start is None:

        def arg_gen(video_ids):
            for video_id in video_ids:
                yield {
                    "video_id": video_id,
                    "output_folder": output_folder,
                    "proxy": proxy,
                    "cookie": cookie,
                }

        arg_iter = arg_gen(video_ids)
    else:

        def arg_gen(video_ids, start_times_list):
            for video_id, start_times in zip(video_ids, start_times_list):
                yield {
                    "video_id": video_id,
                    "start_times": start_times,
                    "output_folder": output_folder,
                    "proxy": proxy,
                    "time_interval": time_interval,
                    "cookie": cookie
                }

        arg_iter = arg_gen(video_ids, start_times_list)

    # for arg in arg_iter:
    #     try:
    #         result = (download_video_process if specify_start is None else download_video_segments_process)(arg)
    #         download_success(result)
    #     except DownloadError as e:
    #         download_fail(e)

    p = Pool(jobs)
    for arg in arg_iter:
        if specify_start is None:
            p.apply_async(
                download_360_process,
                args=(arg,),
                callback=download_success,
                error_callback=download_fail,
            )
        else:
            p.apply_async(
                download_360_segments_process,
                args=(arg,),
                callback=download_success,
                error_callback=download_fail,
            )
    p.close()
    p.join()

    print("Finished downloading.")

    # output fail status
    success_list.sort()
    with open(success_list_name, "w") as f:
        for item in success_list:
            f.write(f"{item}\n")
    print(
        f"{len(success_list)} success files written into {success_list_name}."
    )
    fail_list.sort()
    with open(fail_list_name, "w") as f:
        for item in fail_list:
            f.write(f"{item}\n")
    print(f"{len(fail_list)} fail files written into {fail_list_name}.")
    
    return success_list


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="downloads",
    )
    parser.add_argument(
        "-st",
        "--start-index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-ed",
        "--end-index",
        help="The next index of the last download item",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-p",
        "--proxy",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fail-list-name",
        type=str,
        default="fail_list.txt",
    )
    parser.add_argument(
        "--success-list-name",
        type=str,
        default="success_list.txt",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--specify-start",
        type=str,
        default="multiple",
        choices=["single", "multiple", None],
    )
    parser.add_argument(
        "--time-interval",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    input_file = args.input
    output_folder = args.output
    start_index = args.start_index
    end_index = args.end_index
    fail_list_name = args.fail_list_name
    success_list_name = args.success_list_name
    proxy = None if args.proxy is None else args.proxy.strip()
    jobs = args.jobs
    specify_start = args.specify_start
    time_interval = args.time_interval
    print(f"Using arguments:\n{pprint.pformat(vars(args))}")

    download_list_360(
        input_file=input_file,
        output_folder=output_folder,
        start_index=start_index,
        end_index=end_index,
        proxy=proxy,
        fail_list_name=fail_list_name,
        success_list_name=success_list_name,
        jobs=jobs,
        specify_start=specify_start,
        time_interval=time_interval,
    )
