import os
import pyuac
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DIMENSION = 576
TENSOR_BATCH_SIZE = 128
BATCH_SIZE = 5000
SUPPORTED_IMAGE_FORMATS = (
    ".jpg", ".jpeg", ".png"
)
ALL_SUPPORTED_FORMAT = SUPPORTED_IMAGE_FORMATS + (".pdf",)

save_dir = os.path.join(os.getcwd(), "Index")
map_path = ""
conn = None
next_id = 0
most_recent_mtime = 0
most_recent_fmtime = 0

def search_folder_tree(f_address, root_level=True, bucket=None):
    child_folders = []
    try:
        with os.scandir(f_address) as files:
            for file in files:
                if file.is_file() and file.name.lower().endswith(ALL_SUPPORTED_FORMAT):
                    bucket.append(file.path)
                elif file.is_dir():
                    child_folders.append(file.path)

                if len(bucket) >= BATCH_SIZE:
                    yield bucket[:]
                    bucket.clear() 
    except (PermissionError, OSError, FileNotFoundError):
        return  

    for folder in child_folders:
        yield from search_folder_tree(f_address=folder, root_level=False, bucket=bucket)

    if root_level and bucket:
        yield bucket[:]
        bucket.clear() 

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

def load_image_as_numpy(path):
    from PIL import Image
    import numpy as np
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.uint8)   # shape (H, W, 3)
    except FileNotFoundError:
        print("Can't open Image")
        return None
    return arr

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

def get_image_np_arr(file):
    # NOTICE : Render a page at native resolution or higher to preserve accuracy in sync with its image counterpart
    import pdfium_wrapper, cv2
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
        return (arrays, file)
    else:
        return None

def background_worker():
    while True:
        task = task_queue.get()
        if task is None:
            break  # Clean shutdown
        task()
        task_queue.task_done()

def load_worker_threads(num, worker_threads):
    import queue
    import threading
    global task_queue
    task_queue = queue.Queue() #Starts background thread
    for _ in range(num):
        worker = threading.Thread(target=background_worker, daemon=True)
        worker.start()
        worker_threads.append(worker)

def create_index(nlist=10):
    import faiss
    quantizer = faiss.IndexFlatL2(DIMENSION)
    index = faiss.IndexIVFFlat(quantizer, DIMENSION, nlist, faiss.METRIC_L2)
    return index

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

def add_index_name(name):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO index_names (name) VALUES (?)", (name,))
    # conn.commit()

def get_folder_paths():
    cur = conn.cursor()
    cur.execute("SELECT path FROM folder_paths")
    return [row[0] for row in cur.fetchall()]

def add_folder_path(path):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO folder_paths (path) VALUES (?)", (path,))
    # conn.commit()

def add_file_path(path, id):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO file_paths (key, id) VALUES (?, ?)", (path, id))
    # conn.commit()

def add_index_entry(id, file_path, page):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO index_entries (id, file_path, page) VALUES (?, ?, ?)", (id, file_path, page))
    # conn.commit()

def load_db():
    import sqlite3
    global conn, next_id, most_recent_mtime, most_recent_fmtime, map_path

    os.makedirs(save_dir, exist_ok=True)
    map_path = os.path.join(save_dir, "map_data.db")
    # Create DB file if it doesn't exist (sqlite3.connect will create an empty one automatically)
    first_time = not os.path.exists(map_path)

    conn = sqlite3.connect(map_path)
    cur = conn.cursor()

    if first_time:
        print("Database not found, creating schema...")

        cur.execute("""
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value INTEGER
        )
        """)
        
        cur.execute("""
        CREATE TABLE index_names (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
        """)
        
        cur.execute("""
        CREATE TABLE folder_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE
        )
        """)
        
        cur.execute("""
        CREATE TABLE file_paths (
            key TEXT PRIMARY KEY,
            id INTEGER, 
            FOREIGN KEY(id) REFERENCES index_entries(id)
        )
        """)
        
        cur.execute("""
        CREATE TABLE index_entries (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            mtime REAL,
            page INTEGER
        )
        """)
        
        # Optionally insert default meta values
        cur.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("next_id", 0))
        cur.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("most_recent_mtime", 0))
        cur.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("most_recent_fmtime", 0))

        conn.commit()
    else:
        next_id = int(get_meta("next_id"))
        most_recent_mtime = float(get_meta("most_recent_mtime"))
        most_recent_fmtime = float(get_meta("most_recent_fmtime"))

def main_thread():
    import tkinter as tk
    from tkinter import StringVar, filedialog, Text, Frame
    
    path_count = 0
    search_complete = False
    build_complete = False

    def browse_onPressed(event):
        path = filedialog.askdirectory(title="Select a Folder")
        if path:
            path_value.set(path)

    def wait_build_complete():
        import gc
        nonlocal build_complete
        if build_complete:
            log_text.delete(1.0, 2.0)
            log_text.insert(1.0,"Added " + str(path_count) + " files" + "\n")
            log_text.insert(tk.END, "Setup Complete")
            build_complete = True
            build_button.destroy()
            finish_button.pack(side="top", after=entry_frame)
            gc.collect()
        else:
            root.after(1000, wait_build_complete)

    def init_build_index(folder_path):
        import numpy as np
        import faiss
        import torch
        import time
        import torchvision.transforms as transforms
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        from itertools import chain
        from concurrent.futures import ProcessPoolExecutor
        global next_id, most_recent_fmtime
        nonlocal build_complete, path_count

        load_db()

        # Iterate batches
        index = create_index()
        bucket = []

        # Initialize MobileNetV3
        # Load pretrained model 
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.eval()

        # Remove classifier 
        feature_extractor = torch.nn.Sequential(
            model.features,
            model.avgpool,
            torch.nn.Flatten(1)
        )

        index_is_trained = False

        start_time = time.time()
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        for paths in search_folder_tree(f_address=folder_path, bucket=bucket):
            path_count += len(paths)
            print(len(paths))

            with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
                np_arr_packages = list(filter(None, executor.map(get_image_np_arr, paths)))
            if not np_arr_packages:
                continue
            np_arr_tuple, paths = zip(*np_arr_packages)
            np_arrays =  list(chain.from_iterable(np_arr_tuple))

            # Write entries to SQLite metadata db
            ids = []
            for arr_tuples_list, path in zip(np_arr_tuple, paths):
                # add_file_path(path, next_id)
                for page in range(1,len(arr_tuples_list)+1):
                    ids.append(next_id)
                    add_index_entry(next_id, path, page)
                    next_id += 1

            # Train and add all features to index
            train_batch = []
            id_ptr = 0
            for features in process_np_arrays(np_arrays=np_arrays, feature_extractor=feature_extractor, normalize=normalize):
                if not index_is_trained:
                    train_batch.append(features)
                    if sum(f.shape[0] for f in train_batch) >= 900:
                        # Concatenate first n samples
                        train_data = np.vstack(train_batch)[:900]
                        index.train(train_data)
                        index_is_trained = True

                        full_buf = np.vstack(train_batch)
                        n = full_buf.shape[0]
                        index.add_with_ids(full_buf, np.array(ids[id_ptr:id_ptr+n], dtype=np.int64))
                        id_ptr += n
                        train_batch.clear()
                else:
                    n = features.shape[0]
                    index.add_with_ids(features, np.array(ids[id_ptr:id_ptr+n], dtype=np.int64))
                    id_ptr += n

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.5f} seconds")
        index_name = os.path.basename(folder_path) or "root_index"
        index_path = os.path.join(save_dir, f"{index_name}.index")
        faiss.write_index(index, index_path)

        set_meta(key="next_id", value=next_id)

        add_index_name(index_name)
        add_folder_path(folder_path)

        conn.commit()
        conn.close()

        build_complete = True

    def build_index(event):
            nonlocal path_count, search_complete
            path_count = 0
            search_complete = False
            folder_path = path_value.get()
            if folder_path:
                log_text.insert(1.0, "...Building...Please wait...")
                task_queue.put(item=lambda : init_build_index(folder_path=folder_path))
                wait_build_complete()
                
    root = tk.Tk(screenName="qlens_screen", baseName="qslens_base", className='setup', useTk=1)
    root.title("Index Setup")
    root.resizable(False, False)

    entry_frame = Frame(root)
    path_label = tk.Label(entry_frame, text="Path of Database:")
    path_value = StringVar()
    path_entry = tk.Entry(entry_frame, width=80, textvariable=path_value)
    browse_button = tk.Button(entry_frame, text="Browse")
    build_button = tk.Button(root, text="Build Index")
    finish_button = tk.Button(root, text="Finish", command=root.destroy)
    log_frame = tk.LabelFrame(root, text="Process Log")
    log_text = Text(log_frame, height=3)

    entry_frame.pack(side="top")
    path_label.grid(row=0, column=0)
    path_entry.grid(row=0, column=1)
    browse_button.grid(row=0, column=2)
    build_button.pack(side="top")
    log_frame.pack(side="top", fill="both", expand=True, padx=30, pady=30)
    log_text.pack(fill="both", expand=True)

    browse_button.bind("<Button-1>", browse_onPressed)
    build_button.bind("<Button-1>", build_index)

    root.mainloop()

if __name__ == "__main__":
    import multiprocessing
    if not pyuac.isUserAdmin():
        pyuac.runAsAdmin()
    multiprocessing.freeze_support()
    worker_threads = []
    load_worker_threads(2, worker_threads)
    main_thread()
