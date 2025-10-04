import os
import gc
import sys
import time
import pyuac
import queue
import signal
import psutil
import pystray
import threading
import numpy as np
from PIL import Image
from threading import Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SUPPORTED_IMAGE_FORMATS = (
    ".jpg", ".jpeg", ".png"
)

ALL_SUPPORTED_FORMAT = SUPPORTED_IMAGE_FORMATS + (".pdf",)
TENSOR_BATCH_SIZE = 128

INDEX_DIR = os.path.join(os.getcwd(), "Index")
MAP_PATH = os.path.join(INDEX_DIR, "map_data.db")
WATCHDOG_NAME = "WatchdogService"
WATCHDOG_PATH = os.path.join(os.getcwd(), "WatchdogService.exe")

dimension = 64

conn = None
observer = None
icon = None
indices = {}
index_folder_paths = []

feature_extractor = None
normalize = None

worker_threads = []
lock = Lock()
task_queue = None

asset_dir = os.path.join(os.getcwd(), "assets")
app_icon = Image.open(os.path.join(asset_dir, "ql_icon.png"))

def get_meta(key):
    cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key = ?", (key,))
    row = cur.fetchone()
    return row[0] if row else None  # Returns None if key doesn't exist

def set_meta(key, value):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, str(value)))
    # conn.commit()

def get_index_names():
    cur = conn.cursor()
    cur.execute("SELECT name FROM index_names")
    return [row[0] for row in cur.fetchall()]

def get_folder_paths():
    cur = conn.cursor()
    cur.execute("SELECT path FROM folder_paths")
    return [row[0] for row in cur.fetchall()]

def add_file_path(path, id):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO file_paths (key, id) VALUES (?, ?)", (path, id))
    # conn.commit()

def add_index_entry(id, file_path, mtime, page):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO index_entries (id, file_path, mtime, page) VALUES (?, ?, ?, ?)", (id, file_path, mtime, page))
    # conn.commit()

def load_feature_extractor():
    from torch import nn
    from torchvision.transforms import Normalize
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    global feature_extractor, normalize
    # Load pretrained model 
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()

    # Remove classifier 
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(1)
    )

    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

def load_map_data():
    import sqlite3
    global conn, most_recent_mtime, most_recent_fmtime, index_folder_paths
    try:
        map_exists = os.path.exists(MAP_PATH)
        if map_exists:
            conn = sqlite3.connect(MAP_PATH, check_same_thread=False)
            most_recent_mtime = get_meta("most_recent_mtime")
            most_recent_fmtime = get_meta("most_recent_fmtime")
            for path in get_folder_paths():
                index_folder_paths.append(path)
        print("Done loading data")
        print(most_recent_mtime)
    except Exception as e:
        print(f"Error loading map data: {e}")
        print("Please ensure the map file exists.")
        exit()

def load_index_path(f_path):
    index_name = ""
    parts = os.path.normpath(f_path).split(os.sep)
    for name in get_index_names():
        if name in parts:
            index_name = name
            break
    if index_name == "":
        print("No valid index")
        return

    if INDEX_DIR == "":
        print("Can't find index directory")
        return
    index_path = os.path.join(INDEX_DIR, f"{index_name}.index") 
    return index_path

def load_index(f_path):
    import faiss
    index_path = load_index_path(f_path)

    index = None
    try:
        if index_path in indices:
            index = indices[index_path]
        else:
            index = faiss.read_index(index_path)
            indices[index_path] = index
            print(f"FAISS index loaded from: {index_path}")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("Please ensure the index file exists and is not corrupted.")
        exit()
    return index


def preprocess(tensor, normalize):
    import torch.nn.functional as F
    # Resize from (3,H,W) → (3,224,224)
    tensor = F.interpolate(
        tensor.unsqueeze(0),  # add batch dim → (1,3,H,W)
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)  # back to (3,224,224)

    # Normalize in-place
    tensor = normalize(tensor)
    return tensor

def np_array_to_tensor(arr, normalize):
    import torch
    if arr is not None:
        # Convert to torch tensor, shape (3, H, W), normalize [0,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        # Add batch dimension
        return preprocess(tensor, normalize).unsqueeze(0)
    else:
        print("No Image to transform")
        return None

def process_np_arrays(np_arrays, feature_extractor, normalize):
    import torch
    tensors = []
    for i, np_arr in enumerate(np_arrays, 1):
        tensors.append(np_array_to_tensor(np_arr, normalize))
        if len(tensors) == TENSOR_BATCH_SIZE or i == len(np_arrays):
            tensor_batch = torch.cat(tensors, dim=0)  # shape (B, 3, 224, 224)
            with torch.no_grad():
                feats = feature_extractor(tensor_batch)
            tensors.clear()
            yield feats.numpy().astype('float32')

def get_image_np_arr(file):
    # NOTICE : Render a page at native resolution or higher to preserve accuracy in sync with its image counterpart
    import pdfium_wrapper, cv2
    mtime = 0
    arrays = []
    if file == "":
        return None
    try:
        if file.lower().endswith(SUPPORTED_IMAGE_FORMATS): 
            bgr_arr = cv2.imread(file)
            bgr_arr_re = cv2.resize(bgr_arr, (224, 224), interpolation=cv2.INTER_AREA)
            np_array = cv2.cvtColor(bgr_arr_re, cv2.COLOR_BGR2RGB)
            if np_array is not None:
                arrays.append(np_array)
        elif file.lower().endswith(".pdf"):
            for arr_rgb in pdfium_wrapper.render_doc(file, 0, 0, 96):
                if arr_rgb is not None:
                    # Drop alpha, convert BGRA → RGB
                    np_array = cv2.resize(arr_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                    arrays.append(np_array)          
    except Exception as e:
        print(f"Error with {file}: {e}")
        return None
    if len(arrays) > 0:
        return (arrays, file, mtime)
    else:
        return None

def wait_for_file_ready(path, timeout=3.0, interval=0.1):
    start_time = time.time()
    last_size = -1
    while time.time() - start_time < timeout:
        try:
            size = os.path.getsize(path)
            # Check if size is stable
            if size == last_size:
                # Try opening the file
                with open(path, 'rb') as f:
                    f.read(1)
                return True
            last_size = size
        except (OSError, PermissionError):
            pass
        time.sleep(interval)
    return False  

def register_tree(f_address):
    import faiss
    child_folders = []
    index = load_index(f_address)
    index_path = load_index_path(f_address)
    try:
        with os.scandir(f_address) as files: 
            for file in files:
                if file.is_file() and file.name.lower().endswith(ALL_SUPPORTED_FORMAT):
                    register_files(file.path, single=False, index=index, index_path=index_path)
                elif file.is_dir():
                    child_folders.append(file.path)
    except PermissionError as e:
        print("Permission error")
    except OSError as e:
        print("OSError")
    if len(child_folders) == 0:
        return 
    for folder in child_folders:
        register_tree(f_address = folder)
    faiss.write_index(index, index_path)

def register_files(new_file, single=True, index=None, index_path=""):
    import faiss
    global most_recent_fmtime, most_recent_mtime
    if not wait_for_file_ready(new_file):
        print(new_file + " isn't ready")
        return
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM file_paths WHERE key = ?", (new_file,))
    if cur.fetchone() is None:
        next_id = get_meta("next_id")
        reg_index = None
        reg_index_path = ""
        if single:
            reg_index = load_index(new_file)
            reg_index_path = load_index_path(new_file)
        else:
            reg_index = index
            reg_index_path = index_path
        hash_package = get_image_np_arr(new_file)
        if hash_package:
            img_arrays, filepath, mtime = hash_package    
            with lock:
                if img_arrays: 
                    for features in process_np_arrays(np_arrays=img_arrays, feature_extractor=feature_extractor, normalize=normalize):
                        add_file_path(filepath, next_id)
                        for page, feature_vector in enumerate(features, start=1):
                            reg_index.add_with_ids(feature_vector.reshape(1, -1), np.array([next_id]))
                            add_index_entry(next_id, filepath, mtime, page)
                            next_id += 1
                        set_meta("next_id", next_id)
                        most_recent_mtime = mtime if mtime > most_recent_mtime else most_recent_mtime
                        set_meta("most_recent_mtime", most_recent_mtime)
                        print(f"Added : {new_file}")
                        conn.commit()
                        if single:
                            faiss.write_index(reg_index, reg_index_path)
                        gc.collect()

def background_worker():
    while True:
        task = task_queue.get()
        if task is None:
            break  # Clean shutdown
        task()
        task_queue.task_done()

def load_worker_threads(num):
    global task_queue, worker_threads
    task_queue = queue.Queue() #Starts background thread
    for _ in range(num):
        worker = threading.Thread(target=background_worker, daemon=True)
        worker.start()
        worker_threads.append(worker)

class FileAdditionHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            if not event.src_path.lower().endswith("_tmp.pdf") and event.src_path.lower().endswith(ALL_SUPPORTED_FORMAT):
                task_queue.put(item=lambda : register_files(event.src_path))
        else:
            register_tree(event.src_path)

def load_watchdog():
    global observer
    start_time = time.time()
    event_handler = FileAdditionHandler()
    observer = Observer()
    if index_folder_paths:
        for path in index_folder_paths:
            if os.path.isdir(path):
                    observer.schedule(event_handler, path=path, recursive=True)
    observer.start()
    end_time = time.time()
    print(f"Watch time: {end_time - start_time:.5f} seconds")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def close_watchdog(name="WatchdogService.exe"): 
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if name in " ".join(proc.info['cmdline']):
                proc.terminate()   # or proc.kill()
                print(f"Killed {name} (PID={proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def on_exit(icon, item):
        icon.stop()
        os.kill(os.getpid(), signal.SIGTERM) # terminate process cleanly

def run_tray():
    global icon
    icon = pystray.Icon(
        "Qlen Watchdog Service",
        app_icon,
        menu=pystray.Menu(
            pystray.MenuItem("Exit", on_exit)
        )
    )
    icon.run()  # blocking call

if __name__ == "__main__":
    if not pyuac.isUserAdmin():
        pyuac.runAsAdmin()
        sys.exit(0)
    load_worker_threads(3)
    load_feature_extractor()
    load_map_data()
    task_queue.put(item=lambda : run_tray())
    load_watchdog()
