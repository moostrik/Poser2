import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import random

from watchdog.observers.api import BaseObserver

class imagePathInfo():
    def __init__(self, path: str, images: list[str], usage_count: int) -> None:
        self.path: str = path
        self.images: list[str] = images
        self.usage_count: int = usage_count

ImageDictType = dict[str, imagePathInfo]

SKIP_FILES: int = 2     # a sequence folder starts with the capture and the portrait; the players show the grin

def get_image_paths(folder) -> list[str]:
    image_files: list[str] = [file for file in os.listdir(folder) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    image_files.sort()
    image_files = image_files[SKIP_FILES:]
    return [os.path.join(folder, file) for file in image_files]

# def get_image_folder_dict(images_dir: str, max_folders: int) -> ImageDictType:
#     folders: list[str] = [folder for folder in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, folder))]
#     image_dict: ImageDictType = {}
#     for folder in folders:
#         folder_path: str = os.path.join(images_dir, folder)
#         image_files: list[str] = get_image_paths(folder_path)
#         if len(image_files) > 0:
#             image_dict[folder] = imagePathInfo(folder_path, image_files, 0)
#     sorted_folders: list[str] = sorted(image_dict.keys(), reverse=True)[:max_folders]
#     return {folder: image_dict[folder] for folder in sorted_folders}

def update_image_folder_dict(image_dict: ImageDictType, images_dir: str, max_folders: int) -> bool:
    has_changed = False
    folders: list[str] = [folder for folder in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, folder))]

    # Check if folders in image_dict still exist
    keys_to_delete: list[str] = [key for key in image_dict if key not in folders]
    for key in keys_to_delete:
        del image_dict[key]
        has_changed = True

    # Get the name of the oldest folder in image_dict
    oldest_folder: str | None = min(image_dict.keys()) if image_dict else None
    for folder in folders:
        # Only process folders that are newer than the oldest one in image_dict
        if (oldest_folder is None or folder > oldest_folder) and folder not in image_dict:
            folder_path: str = os.path.join(images_dir, folder)
            image_files: list[str] = get_image_paths(folder_path)
            if len(image_files) > 0:
                image_dict[folder] = imagePathInfo(folder_path, image_files, 0)
                has_changed = True
    sorted_folders: list[str] = sorted(image_dict.keys(), reverse=True)[:max_folders]
    keys_to_delete: list[str] = [key for key in image_dict if key not in sorted_folders]
    for key in keys_to_delete:
        del image_dict[key]
    return has_changed

def get_image_paths_from_dict(image_dict: ImageDictType, max_usage_count: int = 10) -> imagePathInfo:
    min_usage_count: int = min(image_dict[folder].usage_count for folder in image_dict)
    min_usage_folders: list[str] = [folder for folder in image_dict if image_dict[folder].usage_count == min_usage_count]
    selected_folder = random.choice(min_usage_folders)
    image_dict[selected_folder].usage_count = min(image_dict[selected_folder].usage_count + 1, max_usage_count)
    return image_dict[selected_folder]

def get_weighted_image_paths_from_dict(image_dict: ImageDictType, max_usage_count: int = 10) -> imagePathInfo:
    folders: list[str] = list(image_dict.keys())
    usage_counts: list[int] = [image_dict[folder].usage_count for folder in folders]
    # Calculate weights as the inverse of usage counts
    weights: list[float] = [1 / (count + 1) for count in usage_counts]  # Add 1 to avoid division by zero
    selected_folder = random.choices(folders, weights=weights, k=1)[0]
    image_dict[selected_folder].usage_count = min(image_dict[selected_folder].usage_count + 1, max_usage_count)
    return image_dict[selected_folder]


class FolderEventHandler(FileSystemEventHandler):
    def __init__(self, folder_scanner) -> None:
        super().__init__()
        self.folder_scanner = folder_scanner

    def on_any_event(self, event) -> None:
        if event.is_directory:
            if event.event_type == 'created' or event.event_type == 'modified' or event.event_type == 'deleted':
                self.folder_scanner.update_folder_dict()

class FolderScanner:
    def __init__(self, root_folder: str, max_folders: int = 10) -> None:
        self.root_folder: str = root_folder
        self.max_folders: int = max_folders
        self.folder_dict: ImageDictType = {}
        update_image_folder_dict(self.folder_dict, self.root_folder, self.max_folders)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def update_folder_dict(self) -> None:
        with self.lock:
            is_updated: bool = update_image_folder_dict(self.folder_dict, self.root_folder, self.max_folders)
            # if (is_updated):
            #     print ("Updated folder dictionary")
            # else:
            #     print ("folder dictionary not updated")

    def start(self) -> None:
        event_handler = FolderEventHandler(self)
        self.observer: BaseObserver = Observer()
        self.observer.schedule(event_handler, self.root_folder, recursive=False)
        self.observer.start()

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join()

    def get_new_images(self) -> imagePathInfo:
        with self.lock:
            return get_weighted_image_paths_from_dict(self.folder_dict)
