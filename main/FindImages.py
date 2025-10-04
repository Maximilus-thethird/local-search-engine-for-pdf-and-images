import os
import sys
import fitz
import time
import pyuac
import queue
import torch
import psutil
import winreg
import tkinter
import threading
import subprocess
import pdfium_wrapper

from lxml import etree
from tkinter import ttk
from pathlib import Path
from threading import Lock
from ttkthemes import ThemedTk
from ctypes import windll, wintypes, byref
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image, ImageTk, ImageEnhance, UnidentifiedImageError, ImageGrab
from tkinter import filedialog, IntVar, StringVar, Canvas, Frame, CENTER, Menu, TclError

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEBOUNCE_TIME = 400

SUPPORTED_IMAGE_FORMATS = (
    ".jpg", ".jpeg", ".png"
)

ALL_SUPPORTED_FORMAT = SUPPORTED_IMAGE_FORMATS + (".pdf",)

INDEX_DIR = os.path.join(os.getcwd(), "Index")
MAP_PATH = os.path.join(INDEX_DIR, "map_data.db")
WATCHDOG_NAME = "WatchdogService"
WATCHDOG_PATH = os.path.join(os.getcwd(), "WatchdogService.py")

current_tab_index = 0

meta_dict = {}
xml_stringvars = []
root_node = None
root_tree = None
change_stack = []
root_dict = {}

indices = {}

#Variable declaration
image_addresses = []
similarity_scores = []

similarity = 100
image_index = 0
image_size = (900, 650)
search_with_image = False

base_address = ""
folder_address = ""
current_displayed_image = 0
current_base_image = 0
found_photo = None
base_image = None
base_photo = None
display_width = image_size[0]

image_canvas_width = 0
image_canvas_height = 0
base_canvas_width = 0
base_canvas_height = 0

indent = 30

debounce_id = None
pushable = True

conn = None
most_recent_mtime = 0
most_recent_fmtime = 0
index_folder_paths = []
task_queue = None
check_interval = 30 * 60

worker_threads = []
lock = Lock()

observer = None

model = None
feature_extractor = None

asset_dir = ""
not_found_img = None
loading_img = None
search_icon = None
collapse_icon = None
expand_icon = None
add_icon = None
add_tree_icon = None
single_icon = None
sidebyside_icon = None
left_icon = None
right_icon = None
screenshot_icon = None
accept_icon = None
reject_icon = None
loading_photo = None

display_mode = "single"
image_accuracy = 0

#Business logic
import torchvision.transforms as transforms
transform = transforms.Compose([
        transforms.ToTensor(),  # (H,W,3) → (3,H,W), scaled to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def process_np_arrays(np_arr):
    import torch
    tensor = transform(np_arr)              # (3,224,224)
    tensor = tensor.unsqueeze(0)            # add batch dim → (1,3,224,224)
    with torch.no_grad():
        feats = feature_extractor(tensor)   # (1, D)
    return feats.squeeze(0).numpy().astype("float32") 

def load_image_as_numpy(img):
    import numpy as np
    import cv2
    try:
        img = img.convert("RGB")
        img_arr = np.array(img, dtype=np.uint8)   # shape (H, W, 3)
        np_array = cv2.resize(img_arr, (224, 224), interpolation=cv2.INTER_AREA)
    except:
        print("Can't process Image")
        return None
    return np_array

def load_path_as_numpy(path):
    import pdfium_wrapper, cv2
    np_array = None
    try:
        if path.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            bgr_arr = cv2.imread(path)
            bgr_arr_re = cv2.resize(bgr_arr, (224, 224), interpolation=cv2.INTER_AREA)
            np_array = cv2.cvtColor(bgr_arr_re, cv2.COLOR_BGR2RGB)
        elif path.lower().endswith(".pdf"):
            arr_rgb = pdfium_wrapper.render_page(path, 0, 0, 0, 96)
            np_array = cv2.resize(arr_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    except FileNotFoundError:
        print("Can't open Image")
        return None
    return np_array

#LOAD MAP DATA

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

def add_index_entry(id, file_path, mtime, page):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO index_entries (id, file_path, page) VALUES (?, ?, ?)", (id, file_path, page))
    # conn.commit()

def get_index_entry(id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM index_entries WHERE id = ?", (id,))
    data = cur.fetchone()
    if data is None:
        print("NONE")
    return data

def load_map_data():
    import sqlite3
    global conn, index_folder_paths
    try:
        map_exists = os.path.exists(MAP_PATH)
        if map_exists:
            conn = sqlite3.connect(MAP_PATH, check_same_thread=False)
            for path in get_folder_paths():
                index_folder_paths.append(path)
        print("Done loading data")
    except Exception as e:
        print(f"Error loading map data: {e}")
        print("Please ensure the map file exists.")
        exit()

def load_assets():
    global asset_dir, not_found_img, loading_img, search_icon, collapse_icon, expand_icon, add_icon, add_tree_icon, single_icon, sidebyside_icon, left_icon, right_icon, screenshot_icon, accept_icon, reject_icon, loading_photo
    try:
        asset_dir = os.path.join(os.getcwd(), "assets")
        not_found_img = Image.open(os.path.join(asset_dir, "not_found.png"))
        loading_img = Image.open(os.path.join(asset_dir, "loading_image.png"))
        search_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "search_icon.ico")))
        collapse_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "collapse_icon.ico")))
        expand_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "expand_icon.ico")))
        add_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "add_icon.ico")))
        add_tree_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "add_tree_icon.ico")))
        single_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "single_icon.ico")))
        sidebyside_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "sidebyside_icon.ico")))
        left_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "left_icon.ico")))
        right_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "right_icon.ico")))
        screenshot_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "screenshot_icon.ico")))
        accept_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "accept_icon.ico")))
        reject_icon = ImageTk.PhotoImage(Image.open(os.path.join(asset_dir, "reject_icon.ico")))
        loading_photo = ImageTk.PhotoImage(loading_img)
    except FileNotFoundError as e:
        print(f"Image not found: {e}")
    except IOError as e:
        print(f"Error opening image: {e}")

def load_index(f_path):
    import faiss
    global indices
    index_name = ""
    folder_path = Path(f_path)
    for name in get_index_names():
        if name in folder_path.parts:
            index_name = name
            break
    if index_name == "":
        print("No valid index")
        return

    if INDEX_DIR == "":
        print("Can't find index directory")
        return
    index_path = os.path.join(INDEX_DIR, f"{index_name}.index") 

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

def find_similar(b_address, f_address, use_image):
    start_time = time.time()
    if conn:
        import faiss
        import gc

        f_abs = os.path.abspath(f_address)
        f_path = os.path.normpath(f_abs)
        folder_path = Path(f_path)
        addresses = []
        scores = []

        b_np_arr = None
        if use_image:
            b_np_arr = load_image_as_numpy(base_image)
        else:
            b_abs = os.path.abspath(b_address)
            b_path = os.path.normpath(b_abs)
            b_np_arr = load_path_as_numpy(b_path)
        if b_np_arr is None:
            print("Hash is none")
            return ([], [])
        b_features = process_np_arrays(b_np_arr).reshape(1, -1)
        index = load_index(f_path)
        if index:
            index.nprobe = 10
            faiss.omp_set_num_threads(2)
            start_time = time.time()
            D, I = index.search(b_features, k=80)
            end_time = time.time()
            print(f"Search time: {end_time - start_time:.5f} seconds")
            for i, dist in zip(I[0], D[0]):
                try:
                    _, path, _, page = get_index_entry(int(i))
                    print(path)
                    file_path = Path(path)
                    if folder_path in file_path.parents:
                        addresses.append((path, page-1))
                        scores.append(dist)
                except KeyError as e:
                    gc.collect()
                    return ([], [])
            gc.collect()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.5f} seconds")
    return (addresses, scores)

def background_worker():
    while True:
        task = task_queue.get()
        if task is None:
            break  # Clean shutdown
        task()
        task_queue.task_done()

def load_MobileNetV3_small():
    global model, feature_extractor
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()

    # Remove classifier 
    feature_extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(1)
    )

def load_worker_threads(num):
    global task_queue, worker_threads
    task_queue = queue.Queue() #Starts background thread
    for _ in range(num):
        worker = threading.Thread(target=background_worker, daemon=True)
        worker.start()
        worker_threads.append(worker)

# Get the working area (excluding taskbar)
user32 = windll.user32
user32.SetProcessDPIAware()

SPI_GETWORKAREA = 0x0030
rect = wintypes.RECT()
windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, byref(rect), 0)

work_width = rect.right - rect.left
work_height = rect.bottom - rect.top

def run_gui():
    root = ThemedTk(screenName="qlens_screen", baseName="qslens_base", className='QLens', useTk=1, theme="radiance")
    root.title("QLens")
    root.geometry(f"{work_width}x{work_height}+{rect.left}+{rect.top}")
    root.state("zoomed")

    tab_control = ttk.Notebook(root)

    home_tab = ttk.Frame(master=tab_control)
    setting_tab = ttk.Frame(master=tab_control)

    tab_control.add(home_tab, text="Home")
    tab_control.add(setting_tab, text="Settings")

    tab_control.pack(expand=True, fill="both")

    load_assets()

    def on_tab_changed(event):
        global current_tab_index
        current_tab_id = tab_control.select()
        current_tab_index = tab_control.index(current_tab_id)

    tab_control.bind("<<NotebookTabChanged>>", on_tab_changed)


    ##HOME TAB
    #Sidebar class
    class ResizableFrame(Frame):
        def __init__(self, parent, **kwargs):
            super().__init__(parent, **kwargs)
            self.bind("<ButtonPress-1>", self.start_resize)
            self.bind("<B1-Motion>", self.resize)
            self.bind("<ButtonRelease-1>", self.stop_resize)
            self.start_x = 0
            self.init_width = self.winfo_width()
            self.max_x = 900
            self.min_x = 100
            self.line = None

        def start_resize(self, event):
            width = self.winfo_width()
            if event.x > width - 20:
                try:
                    self.start_x = event.x
                    line_y = topbar.winfo_height() + 10
                    line_height = work_height - line_y - 42
                    self.line = Frame(master=home_tab, width=1, height=line_height, bg="black")
                    self.line.place(x=event.x, y=line_y)
                except TclError:
                    pass
        
        def resize(self, event):
            if self.min_x <= event.x <= self.max_x:
                self.line.place(x=event.x)
            elif event.x <= self.min_x:
                self.line.place(x=self.min_x)
            elif event.x >= self.max_x:
                self.line.place(x=self.max_x)

        def stop_resize(self, event):
            final_x = 500
            try:
                final_x = self.line.winfo_x()
            except TclError:
                print("Resize Error")
                return
            if final_x < self.min_x:
                final_x = self.min_x
            elif final_x > self.max_x:
                final_x = self.max_x
            dx = final_x - self.start_x
            new_width = self.winfo_width() + dx
            self.config(width=new_width)
            self.update_idletasks()
            self.line.destroy()

    class VerticalScrolledFrame(Frame):
        def __init__(self, parent, **kwargs):
            super().__init__(parent, **kwargs)

            # Create a canvas object and a vertical scrollbar for scrolling it.
            self.canvas = Canvas(self, bd=0, highlightthickness=0, bg="white", width = 200, height = 300)
            self.canvas.pack(side="left", fill="both", expand=True)

            # Reset the view
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)

            self._scrollable = False

            # Create a frame inside the canvas which will be scrolled with it.
            self.interior = Frame(self.canvas, bg="white")
            self.interior.bind('<Configure>', self._configure_interior)
            self.canvas.bind('<Configure>', self._configure_canvas)
            # self.canvas.bind("<MouseWheel>", lambda e: self._scroll_frame(e=e, what="units"))
            self.interior_id = self.canvas.create_window(0, 0, window=self.interior, anchor="nw")
            self.interior.bind('<Enter>', self._bind_to_mousewheel)
            self.interior.bind('<Leave>', self._unbind_from_mousewheel)

        def _configure_interior(self, event):
            # Update the scrollbars to match the size of the inner frame.
            size = (self.interior.winfo_reqwidth(), self.interior.winfo_reqheight())
            self.canvas.config(scrollregion=(0, 0, size[0], size[1]))
            if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
                # Update the canvas's width to fit the inner frame.
                self.canvas.config(width = self.interior.winfo_reqwidth())
            
        def _configure_canvas(self, event):
            if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
                # Update the inner frame's width to fill the canvas.
                self.canvas.itemconfigure(self.interior_id, width=self.canvas.winfo_width())

        def _scroll_frame(self, e, what="units"):
            if self._scrollable:
                self.canvas.yview_scroll(int(-1*(e.delta/120)), what)

        def _get_scrollable(self):
            return self._scrollable

        def _set_scrollable(self, scrollable):
            self._scrollable = scrollable

        def _bind_to_mousewheel(self, event):
            self.canvas.bind_all("<MouseWheel>", lambda e: self._scroll_frame(e=e, what="units"))

        def _unbind_from_mousewheel(self, event):
            self.canvas.unbind_all("<MouseWheel>")

        def _update_scrollable(self):
            self.canvas.update_idletasks()
            scrollregion = self.canvas.bbox("all")

            if scrollregion:
                scroll_height = scrollregion[3] - scrollregion[1]
                visible_height = self.canvas.winfo_height()

                self._set_scrollable(scroll_height > visible_height)

    #Element frame class
    class ElementFrame(Frame):
        def __init__(self, master=None, node=None, level=0, **kwargs):
            global root_dict
            super().__init__(master, **kwargs)
            self.__node = node
            self.__path = ""
            self.__toggle = False
            self.__has_typed = False
            self.__has_spaced = False
            self.__is_deleting = False
            self.__skip_delete_in_apply = False
            self.__level = level
            self.__children = []
            self.__buffer_str_onClicked = ""
            self.__bypass = False

            self.pack_propagate(True)

            self.__field_frame = Frame(master=self)
            self.__field_frame.grid_columnconfigure(0, minsize=29)

            self.__collapse_button = ttk.Button(master=self.__field_frame, image=collapse_icon, command=self.toggle_collapse)
            self.__add_property_button = ttk.Button(master=self.__field_frame, image=add_icon, command=self.add_property)
            self.__add_tree_button = ttk.Button(master=self.__field_frame, image=add_tree_icon, command=self.add_tree)
            self.__label = ttk.Label(master=self.__field_frame, text="") # Default empty
            self.__xml_value = StringVar()
            self.__xml_textfield = ttk.Entry(master=self.__field_frame, width=work_width - 100, textvariable=self.__xml_value)
            self.__attr_label = ttk.Label(master=self.__field_frame, text="") # For attributes

            self.__label.bind("<Button-3>",  self.show_change_meta_menu)
            self.__xml_textfield.bind("<Button-3>", self.show_change_meta_menu)
            self.__xml_textfield.bind("<Button-1>", self.on_clicked)
            self.__xml_textfield.bind("<BackSpace>", self.backspace_onPressed)
            self.__add_property_button.bind("<Button-3>", self.show_change_meta_menu)
            self.__add_tree_button.bind("<Button-3>", self.show_change_meta_menu)

            if node is not None:
                for child in self.__node:
                    self.__children.append(ElementFrame(master=self, node=child, level=self.__level+1))
                self.__label.config(text=self.__node.tag + ":")

        @staticmethod 
        def is_valid_start(char):
            import unicodedata
            category = unicodedata.category(char)
            return category.startswith('L') 
        
        @staticmethod 
        def sanitize_xml_tag(tag_name: str) -> str:
            import re
            # Replace spaces and invalid characters with underscores
            tag = re.sub(r'[^\w.-]', '_', tag_name.strip(), flags=re.UNICODE)

            if not ElementFrame.is_valid_start(tag[0]):
                tag = f"tag_{tag}"

            return tag
        
        @staticmethod
        def sanitize_xml_value(text: str) -> str:
            import re
            text = re.sub(r'[\ud800-\udfff]', '', text)
            text = text.replace('\uFFFD', '')
            return text

        @staticmethod
        def strip_rdf_namespaces(xml_metadata):
            for elem in xml_metadata.iter():
                if not isinstance(elem.tag, str):
                    continue
                elem.tag = etree.QName(elem).localname  # Remove namespace prefix
            etree.cleanup_namespaces(xml_metadata)

        def display(self, before=None):
            if before:
                self.pack(anchor="w", pady=2, before=before)
            else:
                self.pack(anchor="w", pady=2)

            if self.__node is not None:
                self.__add_property_button.grid(row=0, column=1)

                self.__add_tree_button.grid(row=0, column=2)

                self.__field_frame.pack(anchor="w", pady=2, padx=(self.__level*indent,0))

                self.__label.grid(row=0, column=3)

                if self.__node.attrib:
                    attr_text = ', '.join(f'{k}="{v}"' for k, v in self.__node.attrib.items())
                    self.__attr_label.config(text=f"[{attr_text}]")
                    self.__attr_label.grid(row=0, column=4)

                text_content = self.__node.text.strip() if self.__node.text and self.__node.text.strip() else ""
                self.__xml_value.set(text_content)
                self.__buffer_str_onTyped = text_content
                xml_stringvars.append(self.__xml_value)
                self.__xml_textfield.grid(row=0, column=5)

                self.__xml_value.trace_add("write", lambda var, index, mode: self.value_onChanged())

                if len(self.__children) != 0:
                    self.__collapse_button.grid(row=0, column=0)
                    for child in self.__children:
                        child.display()

        def get_parent_node(self):
            parent_name = self.winfo_parent()
            parent = self.nametowidget(parent_name)
            if parent is default_frame:
                return None
            return parent.get_node()

        def get_node(self):
            return self.__node

        def set_node(self, node):
            self.__node = node
            self.__children = []
            if self.__node is not None:
                for child in self.__node:
                    self.__children.append(ElementFrame(master=self, node=child, level=self.__level+1))
                self.__label.config(text=self.__node.tag + ":")

        def set_bypass(self, bypass):
            self.__bypass = bypass

        def get_stringvar(self):
            return self.__xml_value

        def get_toggle(self):
            return self.__toggle

        def set_toggle(self, toggle: bool):
            self.__toggle = toggle

        def get_children(self):
            return self.__children
        
        def get_collapse_button(self):
            return self.__collapse_button
        
        def set_path(self):
            global root_dict
            self.__path = root_tree.getpath(self.__node)
            root_dict[self.__path] = self
            if len(self.__children) !=0:
                for child in self.__children:
                    child.set_path()
        
        def get_path(self):
            return self.__path
        
        def get_entry(self):
            return self.__xml_textfield
            
        def toggle_collapse(self):
            if self.__toggle: #Is collapsed
                for child in self.__children:
                    child.pack(anchor="w", pady=2)
                self.__toggle = False
                self.__collapse_button.config(image=collapse_icon)
            else: #Is NOT collapsed
                for child in self.__children:
                    child.pack_forget()
                self.__toggle = True
                self.__collapse_button.config(image=expand_icon)
            sidebar_content_frame._update_scrollable()

        def push_change_to_stack(self, action, node_frame, value="", index=0, cursor=0):
            parent_path = "default"
            if node_frame.nametowidget(node_frame.winfo_parent()) is not default_frame:
                parent_path = node_frame.nametowidget(node_frame.winfo_parent()).get_path()

            change_stack.append({
                "action" : action,
                "node_path" : node_frame.get_path(),
                "parent_path" : parent_path,
                "node_frame_copy" : node_frame,
                "value" : value,
                "index" : index,
                "cursor" : cursor
            })

        def copy(self, parent=None):
            clone = None
            if parent:
                clone = ElementFrame(master=parent, node=self.__node, level=self.__level)
            else:
                self_parent_name = self.winfo_parent()
                self_parent = self.nametowidget(self_parent_name)
                clone = ElementFrame(master=self_parent, node=self.__node, level=self.__level)
            return clone

        def set_has_typed(self, has_typed):
            self.__has_typed = has_typed

        def set_has_spaced(self, has_spaced):
            self.has_spaced = has_spaced

        def set_is_deleting(self, is_deleting):
            self.__is_deleting = is_deleting

        def add_child(self, child_frame):
            child_node = child_frame.get_node()
            self.__node.append(child_node)
            self.__children.append(child_frame)
            child_frame.set_path()

        def insert_child(self, index, child):
            if index >= 0 and index <= len(self.__children):
                self.__children.insert(index, child)
                self.__node.insert(index, child.get_node())
                child.set_path()

        def add_property_onPressed(self, event, add_parent, title):
            global root_tree, root_node
            add_parent.destroy()
            add_parent = None
            if not title:
                self.config(width=1, height=1)
                sidebar_content_frame._update_scrollable()
                return
            if insert_frame:
                insert_frame.destroy()

            new_element = None
            try:
                new_element = etree.Element(title)
            except ValueError:
                title = ElementFrame.sanitize_xml_tag(title)
                new_element = etree.Element(title)

            if self.__node is not None:
                next_level = self.__level + 1
                new_element_frame = ElementFrame(master=self, node=new_element, level=next_level)
                self.add_child(new_element_frame)
                print(new_element_frame.get_path())
                self.__collapse_button.grid(row=0, column=0)
                new_element_frame.display()
                task_queue.put(item=lambda : self.push_change_to_stack(action="add", node_frame=new_element_frame))
            else:
                insert_frame.destroy()
                self.set_node(new_element)
                root_node = new_element
                root_tree = etree.ElementTree(self.__node)
                self.set_path()
                self.display()
                task_queue.put(item=lambda : self.push_change_to_stack(action="add", node_frame=self))
            sidebar_content_frame._update_scrollable()
        
        def add_property(self):
            if self.__toggle:
                self.toggle_collapse()
            add_property_frame = Frame(master=self)
            new_key = StringVar()
            xml_stringvars.append(new_key)
            meta_label_entry = ttk.Entry(master=add_property_frame, textvariable= new_key)
            meta_ok_button = ttk.Button(master=add_property_frame, text="OK")
            meta_cancel_button = ttk.Button(master=add_property_frame, text="Cancel")

            next_level = 0
            if self.__node is not None:
                next_level = self.__level + 2 #The reason it's 2 because we have to take in account the presence of the collapse button
            add_property_frame.pack(anchor="w", padx=(next_level * indent,0), pady=2)
            meta_label_entry.grid(row=0, column=0)
            meta_ok_button.grid(row=0, column=1)
            meta_cancel_button.grid(row=0, column=2)
            
            meta_ok_button.bind("<Button-1>", func=lambda event: self.add_property_onPressed(event=event, add_parent=add_property_frame, title=new_key.get()))
            meta_cancel_button.bind("<Button-1>", func=lambda event: self.add_property_onPressed(event=event, add_parent=add_property_frame, title=""))

            sidebar_content_frame._update_scrollable()

        def add_tree(self):
            from lxml import etree
            global root_node, root_tree
            xml_path = filedialog.askopenfilename(
                title="Select a File",
                filetypes=[("XML files", "*.xml")]
            )
            if xml_path: 
                if self.__toggle:
                    self.toggle_collapse()
                try:
                    tree = etree.parse(xml_path)
                    if self.__node is None:
                        root_node = tree.getroot()
                        root_tree = etree.ElementTree(root_node)
                        insert_frame.destroy()
                        ElementFrame.strip_rdf_namespaces(root_node)
                        self.set_node(root_node)
                        self.set_path()
                        self.display()
                        task_queue.put(item=lambda : self.push_change_to_stack(action="add", node_frame=self))
                        sidebar_content_frame._update_scrollable()
                    else:
                        new_node = tree.getroot()
                        ElementFrame.strip_rdf_namespaces(new_node)
                        new_node_frame = ElementFrame(master=self, node=new_node, level=self.__level+1)
                        self.add_child(new_node_frame)
                        new_node_frame.display()
                        if len(self.__children) != 0:
                            self.__collapse_button.grid(row=0, column=0)
                        task_queue.put(item=lambda : self.push_change_to_stack(action="add", node_frame=new_node_frame))
                        sidebar_content_frame._update_scrollable()
                except etree.XMLSyntaxError as e:
                    print(f"XML is not well-formed: {e}")
                except FileNotFoundError:
                    print("The file was not found.")
                except PermissionError:
                    print("Permission denied. Cannot open file.")
                except Exception as e:
                    print(f"Unexpected error: {e}")
            else:
                return
            
        def remove_child(self, child):
            global root_dict
            self.__node.remove(child.get_node())
            try:
                self.__children.remove(child)
            except ValueError:
                print("No child to remove")
            child.destroy()
            child.remove_path()

        def remove_path(self):
            global root_dict
            root_dict.pop(self.__path, None)
            if len(self.__children) != 0:
                for child in self.__children:
                    child.remove_path()

        def add_path_to_dict(self):
            global root_dict, root_tree
            root_dict[self.__path] = self
            if len(self.__children) != 0:
                for child in self.__children:
                    child.add_path_to_dict()

        def delete_self(self):
            global root_node, root_node_frame, root_dict
            parent_frame = self.nametowidget(self.winfo_parent())
            self_copy = self.copy()
            parent_node = self.__node.getparent()
            index = 0
            if parent_node is not None:
                index = parent_node.index(self.__node)
            if parent_frame is not default_frame:
                parent_frame.remove_child(child=self)
                if len(parent_frame.get_children()) == 0:
                    parent_frame.get_collapse_button().grid_forget()
            else:
                root_node = None
                self.destroy()
                root_node_frame = None
                display_insertbar(parent=default_frame)
                self.remove_path()

            task_queue.put(item=lambda : self.push_change_to_stack(action="delete", node_frame=self_copy, index=index))
            sidebar_content_frame._update_scrollable()

        def set_skip_apply_push(self, skip):
            self.__skip_delete_in_apply = skip
        
        def on_clicked(self, event):
            self.__buffer_str_onClicked = self.__xml_value.get()
            self.__has_typed = False
            self.__has_spaced = False
            self.__is_deleting = False
            self.__skip_delete_in_apply = False

        def backspace_onPressed(self, event): #DONT FUCKING TOUCH THIS EVER AGAIN
            if self.__node.text:
                if not self.__is_deleting:
                    node_text_len = len(self.__node.text) if self.__node.text else 0
                    cursor = node_text_len - len(self.__xml_value.get()) + self.__xml_textfield.index(tkinter.INSERT)
                    self.push_change_to_stack(action="modify", node_frame=self, value=self.__node.text, cursor=cursor)
                    self.__is_deleting = True
                    self.__skip_delete_in_apply = True
            
        def show_change_meta_menu(self, event):
            change_meta_menu.entryconfigure(0, command=lambda: self.delete_self())
            change_meta_menu.post(event.x_root, event.y_root)

        def apply_value_change(self): #DONT FUCKING TOUCH THIS EVER AGAIN
            text_value = self.__xml_value.get()
            text_value = ElementFrame.sanitize_xml_value(text_value)
            prev_len = len(self.__node.text) if self.__node.text else 0
            if len(text_value) > prev_len:
                self.__is_deleting = False
                self.__skip_delete_in_apply = False
            self.__node.text = text_value
            current_cursor = self.__xml_textfield.index(tkinter.INSERT)
            if not self.__skip_delete_in_apply:
                if text_value:
                    if not self.__has_spaced:
                        if text_value[current_cursor - 1] == " ":
                            cursor = current_cursor - 1
                            self.push_change_to_stack(action="modify", node_frame=self, value=text_value[:cursor] + text_value[cursor+1:], cursor=cursor)
                            self.__has_spaced = True
                    elif text_value[current_cursor-1] != " ":
                        self.__has_spaced = False
                if not self.__has_typed:
                    cursor = len(self.__buffer_str_onClicked) - len(text_value) + current_cursor
                    self.push_change_to_stack(action="modify", node_frame=self, value=self.__buffer_str_onClicked, cursor=cursor)
                    self.__has_typed = True

        def value_onChanged(self):
            if not self.__bypass:
                task_queue.put(self.apply_value_change)
        
    #Image processing
    class ScreenCapture:
        def __init__(self):
            root.withdraw()
            time.sleep(0.5)
            self.root = tkinter.Toplevel()

            self.root.attributes("-fullscreen", True)

            self.start_x = self.start_y = None
            self.x1 = self.y1 = self.x2 = self.y2 = 0
            self.rect = None
            self.image = None
            self.bg_img = ImageGrab.grab()
            self.bg_dimmed = ImageEnhance.Brightness(self.bg_img).enhance(0.5)
            self.bg_dimmed_photo = ImageTk.PhotoImage(self.bg_dimmed)
            self.canvas = Canvas(self.root, cursor="cross")
            self.canvas.pack(fill=tkinter.BOTH, expand=True)

            self.canvas.create_image(0, 0, image=self.bg_dimmed_photo, anchor="nw")

            self.command_frame = ttk.Frame(self.canvas)
            self.accept_button = ttk.Button(self.command_frame, image=accept_icon, command=self.capture_area)
            self.reject_button = ttk.Button(self.command_frame, image=reject_icon, command=self.close_area)

            self.accept_button.grid(row=0, column=0)
            self.reject_button.grid(row=0, column=1)

            self.info_label = ttk.Label(self.canvas, text="Press Esc to cancel")
            self.info_label.place(x=0, y=0)

            self.canvas.bind("<ButtonPress-1>", self.on_press)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)

            self.root.bind_all("<Escape>", self.exit_window)
            self.root.focus_force()

            self.root.mainloop()

        def exit_window(self, event=None):
            self.close_area()
            self.root.destroy()
            self.clear_mem()
            root.deiconify()
            root.geometry(f"{work_width}x{work_height}+{rect.left}+{rect.top}")
            root.state("zoomed")

        def on_press(self, event):
            self.info_label.place_forget()
            self.close_area()
            self.start_x, self.start_y = event.x, event.y
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y,
                                                    self.start_x, self.start_y,
                                                    outline="white", width=3, dash=(10,5))

        def on_drag(self, event):
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

        def on_release(self, event):
            self.x1, self.y1, self.x2, self.y2 = self.start_x, self.start_y, event.x, event.y
            if self.x1 != self.x2 and self.y1 != self.y2:
                self.command_frame.place(x=self.start_x, y=self.start_y)
            # self.capture_area(x1, y1, x2, y2)

        def clear_mem(self):
            import gc
            del self.bg_dimmed_photo
            self.canvas.destroy()
            self.command_frame.destroy()
            self.accept_button.destroy()
            self.reject_button.destroy()
            self.info_label.destroy()
            gc.collect()

        def capture_area(self):
            global base_image, search_with_image
        
            self.root.attributes("-alpha", 0)
            self.image = self.bg_img.crop((self.x1, self.y1, self.x2, self.y2))
            # self.root.destroy()
            # img.show()  # for testing
            base_image = self.image
            # root.deiconify()
            self.exit_window()
            display_base_image(use_address=False)
            search_with_image = True
            base_address_value.set("")

        def close_area(self):
            self.start_x = self.start_y = None
            self.x1 = self.y1 = self.x2 = self.y2 = 0
            self.canvas.delete(self.rect)
            self.image = None
            self.command_frame.place_forget()

        def get_image(self):
            return self.image

    def resize_image(image, canvas):
        image_width, image_height = image.size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        width_ratio = canvas_width / image_width
        height_ratio = canvas_height / image_height
        scale_factor = min(width_ratio, height_ratio)
        return image.resize((int(image_width * scale_factor), int(image_height * scale_factor)), Image.LANCZOS)

    def display_insertbar(parent):
        global insert_frame, root_node_frame, root_tree
        insert_frame = Frame(master=parent)
        insert_new_button = ttk.Button(master=insert_frame, text="Insert new element")
        insert_tree_button = ttk.Button(master=insert_frame, text="Insert new tree")
        
        insert_frame.pack(anchor="w", pady=2)
        insert_new_button.pack(side="left")
        insert_tree_button.pack(side="left")
        if root_node_frame is None:
            root_node_frame = ElementFrame(master=default_frame, node=root_node)
            root_tree = etree.ElementTree(root_node)
        root_node_frame.display()
        insert_new_button.bind("<Button-1>", func=lambda event: root_node_frame.add_property())
        insert_tree_button.bind("<Button-1>", func=lambda event: root_node_frame.add_tree())
        
    def display_metadata(pdf_metadata):
        for key, value in pdf_metadata.items():
            field_frame = Frame(master=sidebar_content_frame.interior)
            field_label = ttk.Label(master=field_frame, text=key)
            meta_textfield_value = StringVar()
            meta_textfield_value.set(value)
            meta_textfield = ttk.Entry(master=field_frame, width=work_width-100, textvariable=meta_textfield_value)
            field_frame.pack(padx=5, pady=2)
            field_label.pack(side="left") 
            meta_textfield.pack(side="left")
            meta_dict[key] = meta_textfield_value

    def parse_metadata(address, yview=0):
        from PIL.ExifTags import TAGS
        from lxml import etree
        global root_node, root_tree, root_node_frame, default_frame, insert_frame, meta_dict, change_stack, xml_stringvars
        root_node = None
        root_tree = None
        root_node_frame = None
        xml_metadata = None
        for widget in sidebar_content_frame.interior.winfo_children():
            widget.destroy()
        for var in list(meta_dict.values()) + xml_stringvars:
            try:
                if var.trace_info():
                    var.trace_remove("write", var.trace_info()[0][1])
            except:
                pass  # Safe cleanup
        meta_dict.clear()
        xml_stringvars.clear()
        change_stack.clear()
        sidebar_content_frame.canvas.yview_moveto(yview) #Important to reset the view when changing page

        if address.endswith(".pdf"):
            pdf_file = None
            try:
                pdf_file = fitz.open(address)
                pdf_metadata = pdf_file.metadata
                xml_metadata = pdf_file.get_xml_metadata()
                if pdf_metadata:
                    display_metadata(pdf_metadata)
            except Exception as e:
                print(f"Fatal error during PDF processing: {e}")
            finally:
                if pdf_file:
                    pdf_file.close()
            
            custom_property_frame = Frame(master=sidebar_content_frame.interior)
            custom_label = ttk.Label(master=custom_property_frame, text="Custom Property:")
            saveex_frame = Frame(master=custom_property_frame) 
            save_property_button = ttk.Button(master=saveex_frame, text="Save")
            export_button = ttk.Button(master=saveex_frame, text="Export")

            custom_property_frame.columnconfigure(0, weight=1)  
            custom_property_frame.columnconfigure(1, weight=1)
            custom_property_frame.columnconfigure(2, weight=1) 

            custom_property_frame.pack(fill="x", expand=True)
            custom_label.grid(row=0, column=1)
            saveex_frame.grid(row=0, column=2, sticky="e")
            save_property_button.grid(row=0, column=0)
            export_button.grid(row=0, column=1)
            save_property_button.bind("<Button-1>", save_onPressed)
            export_button.bind("<Button-1>", export_onPressed)
            default_frame = ttk.Frame(master=sidebar_content_frame.interior)
            default_frame.pack(anchor="w", pady=2)
            default_frame.pack_propagate(True)

            insert_frame = None

            if xml_metadata:
                try:
                    root_node = etree.fromstring(xml_metadata)
                    ElementFrame.strip_rdf_namespaces(root_node) #Clean up data
                    root_tree = etree.ElementTree(root_node)
                    root_node_frame = ElementFrame(master=default_frame, node=root_node)
                    root_node_frame.set_path()
                    root_node_frame.display()
                except etree.XMLSyntaxError:
                    syntax_error_label = ttk.Label(master=sidebar_content_frame.interior, text="XML Syntax Error!")
                    syntax_error_label.pack()
                    print("XMLSyntaxError")
                except Exception as e:
                    print(f"Unexpected XML-related error: {e}")
            else:
                display_insertbar(parent=default_frame)

        elif address.endswith(SUPPORTED_IMAGE_FORMATS):
            image_file = Image.open(address)
            exif_data = image_file._getexif()
            image_file.close()
            if exif_data:
                for id, value in exif_data.items():
                    key_name = TAGS.get(id, id)
                    field_frame = Frame(master=sidebar_content_frame.interior)
                    field_label = ttk.Label(master=field_frame, text=key_name)
                    meta_textfield_value = StringVar()
                    if isinstance(value, bytes):
                        display_value = value.decode(errors="ignore")
                    elif isinstance(value, tuple):
                        display_value = ", ".join(map(str, value))
                    else:
                        display_value = str(value)
                    meta_textfield_value.set(display_value)
                    meta_textfield = ttk.Entry(master=field_frame, textvariable=meta_textfield_value, width=70, state="disabled")
                    field_frame.pack(padx=5, pady=2)
                    field_label.grid(row=0, column=0) 
                    meta_textfield.grid(row=0, column=1)
                    meta_dict[key_name] = meta_textfield_value

        sidebar_content_frame._update_scrollable()

    # Event Handlers 
    def revert_onPressed(event):
        global root_node, change_stack, root_node_frame, root_dict, root_tree
        # task_queue.join()
        if change_stack and default_frame:
            change_event = change_stack.pop()
            match change_event["action"]:
                case "add":
                    node = root_dict[change_event["node_path"]]
                    if change_event["parent_path"] != "default":
                        parent_frame = root_dict[change_event["parent_path"]] #Basically delete_self but without pushing change to stack
                        parent_frame.remove_child(child=node)
                        if len(parent_frame.get_children()) == 0:
                            parent_frame.get_collapse_button().grid_forget()
                    else:
                        root_node = None
                        node.destroy()
                        root_node_frame = None
                        display_insertbar(parent=default_frame)
                    root_dict.pop(change_event["node_path"], None)
                case "delete":
                    node = change_event["node_frame_copy"]
                    if change_event["parent_path"] != "default":
                        parent_frame = root_dict[change_event["parent_path"]]
                        node_copy = node.copy(parent=parent_frame)
                        parent_frame.insert_child(change_event["index"], node_copy)
                        node_copy.add_path_to_dict() #Add all the removed paths back to dict, including paths of children
                        len_of_children = len(parent_frame.get_children())
                        if len_of_children > 1 and change_event["index"] < len_of_children - 1:
                            right_side_child = parent_frame.get_children()[change_event["index"]+1]
                            node_copy.display(before=right_side_child)
                        else:
                            node_copy.display()
                    else:
                        insert_frame.destroy()
                        root_node_frame = node
                        root_node = node.get_node()
                        root_tree = etree.ElementTree(root_node)
                        root_node_frame.set_path()
                        node.add_path_to_dict() #Add all the removed paths back to dict, including paths of children
                        root_node_frame.display()
                case "modify":
                    node = root_dict[change_event["node_path"]]
                    node.set_bypass(bypass=True)
                    node.get_stringvar().set(change_event["value"])
                    node.get_node().text = change_event["value"]
                    node.set_bypass(bypass=False)
                    current_textfield = node.get_entry()
                    if root.focus_get() is not current_textfield:
                        current_textfield.focus_set()
                    current_textfield.icursor(change_event["cursor"])
                    node.set_has_typed(has_typed=False)
                    node.set_has_spaced(has_spaced=False)
                    node.set_is_deleting(is_deleting=False)
                    node.set_skip_apply_push(skip=False)
            sidebar_content_frame._update_scrollable()

    def save_onPressed(event):
        from lxml import etree
        global temp_path
        if len(image_addresses) > 0:
            if image_addresses[image_index-1][0].endswith(".pdf"):
                pdf_file = fitz.open(image_addresses[image_index-1][0])
                if root_node is not None:
                    xml_str = etree.tostring(root_node, encoding='utf-8').decode('utf-8')
                    pdf_file.set_xml_metadata(xml_str)
                else:
                    pdf_file.set_xml_metadata("")
                temp_path = image_addresses[image_index-1][0].replace(".pdf", "_tmp.pdf")
                pdf_file.save(temp_path)
                pdf_file.close()
                os.replace(temp_path, image_addresses[image_index-1][0])

    def export_onPressed(event):
        from lxml import etree
        if root_node is None:
            print("No root to export")
            return
        else:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xml",
                filetypes=[("XML Files", "*.xml"), ("All Files", "*.*")],
                title="Save As XML"
            )
            if file_path:
                tree = etree.ElementTree(root_node)
                try:
                    tree.write(file_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
                    print(f"Successfully saved XML to: {file_path}")
                except PermissionError as perm_error:
                    print(f"Permission denied: {perm_error}")
                except (OSError, IOError) as file_error:
                    print(f"File I/O error: {file_error}")
                except etree.SerialisationError as ser_error:
                    print(f"Serialization error: {ser_error}")
                except Exception as e:
                    print(f"Unexpected error: {e}")
            else:
                return
            
    def display_base_image(use_address=True):
        global base_image
        if use_address:
            try:
                if base_address.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                    with Image.open(base_address) as img:
                        base_image = img.copy()
                elif base_address.lower().endswith(".pdf"):
                    base_image = pdfium_wrapper.convert_page_to_image(base_address, 0, 0, 0, 80)
            except (OSError, FileNotFoundError, ValueError, Exception) as e:
                base_image = not_found_img
                print("Can't open file")
                print(e)
        if display_mode != "sidebyside":
            switch_to_sidebyside()
        else:
            root.after(50, wait_until_base_canvas_ready_and_draw())

    def base_onPressed(event):
        global base_address, base_image, search_with_image
        address = filedialog.askopenfilename(
            title="Select a File",
            filetypes=[("Images & PDFs", ALL_SUPPORTED_FORMAT), ("All Files", "*.*")]
        )
        if address:
            base_address_value.set(address)
            base_address = address
            display_base_image()
        search_with_image = False

    def folder_onPressed(event):
        global folder_address
        address = filedialog.askdirectory(title="Select a Folder")
        if address:
            folder_address_value.set(address)
            folder_address = address

    def ok_onPressed(event):
        global image_index, current_displayed_image, found_photo, image_addresses, similarity_scores, base_address
        image_addresses.clear()
        found_photo = None

        if current_displayed_image == 0:
            current_displayed_image = image_canvas.create_image(image_canvas_width/2, image_canvas_height/2, anchor = CENTER, image=loading_photo)
        else:
            image_canvas.itemconfig(current_displayed_image, image=loading_photo)

        image_canvas.update_idletasks()
        base_field_address = base_address_value.get()
        folder_field_address = folder_address_value.get()
        if os.path.isfile(base_field_address) and os.path.isdir(folder_field_address):
            image_addresses, similarity_scores = find_similar(base_field_address, folder_field_address, False)
        elif os.path.isdir(folder_field_address) and search_with_image:
            image_addresses, similarity_scores = find_similar(base_field_address, folder_field_address, True)
        else:
            image_canvas.delete(current_displayed_image)
            current_displayed_image = 0
            return 
        image_index = 1
        match_address = image_addresses[image_index-1][0]
        match_page = image_addresses[image_index-1][1]
        number_of_addresses = len(image_addresses)
        if number_of_addresses != 0:
            found_image = None
            found_photo = None
            try:
                if match_address.lower().endswith(".pdf"):
                    found_image = pdfium_wrapper.convert_page_to_image(match_address, match_page, 0, 0, 80)
                else:
                    found_image = Image.open(match_address)
                if found_image == None or found_image.width <= 0 or found_image.height <= 0:
                    found_photo = ImageTk.PhotoImage(resize_image(not_found_img, image_canvas))
                    image_canvas.itemconfig(current_displayed_image, image = found_photo)
                    score_label.config(text=" ")
                    image_counter_label.config(text=f"  {0} / {0}  ")
                    image_address_var.set(match_address + f"---PAGE : {match_page + 1}")
                    return
                found_photo = ImageTk.PhotoImage(resize_image(found_image, image_canvas))
                found_image.close()
            except (OSError, Image.UnidentifiedImageError, ValueError, Exception) as e:
                found_photo = ImageTk.PhotoImage(resize_image(not_found_img, image_canvas))
                print(f"Can't open file: {e}")

            image_canvas.itemconfig(current_displayed_image, image = found_photo)
            score_label.config(text=f"Distance : {similarity_scores[image_index - 1]}")
            image_counter_label.config(text=f"  {image_index} / {number_of_addresses}  ")
            image_address_var.set(match_address + f"---PAGE : {match_page + 1}")

            parse_metadata(match_address)
        else:
            if current_displayed_image != 0:
                image_canvas.delete("all")
                current_displayed_image = 0
            else:
                found_photo = ImageTk.PhotoImage(resize_image(not_found_img, image_canvas))
                image_canvas.itemconfig(current_displayed_image, image = found_photo)
            score_label.config(text=" ")
            image_counter_label.config(text="  0 / 0  ")
            image_address_var.set("")
            current_displayed_image = 0
            for widget in sidebar_content_frame.interior.winfo_children():
                widget.destroy()
            print("true")

    def page_transit(index):
        global found_photo
        found_image = None
        match_address = image_addresses[index-1][0]
        match_page = image_addresses[index-1][1]
        try:
            if match_address.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                with Image.open(match_address) as img:
                    found_image = img.copy()
            elif match_address.lower().endswith(".pdf"):
                found_image = pdfium_wrapper.convert_page_to_image(match_address, match_page, 0, 0, 80)
            found_photo = ImageTk.PhotoImage(resize_image(found_image, image_canvas))
        except (OSError, FileNotFoundError, ValueError, Exception) as e:
            found_photo = ImageTk.PhotoImage(resize_image(not_found_img, image_canvas))
            print(f"Can't open file: {e}")

        image_canvas.itemconfig(current_displayed_image, image=found_photo)
        score_label.config(text=f"Distance : {similarity_scores[index - 1]}")
        image_counter_label.config(text=f"  {index} / {len(image_addresses)}  ")
        image_address_var.set(match_address + f"---PAGE : {match_page+1}")
        # parse_metadata(match_address)

    def left_onPressed(event):
        global image_index, current_displayed_image
        if image_index > 1:
            image_index -= 1
            page_transit(image_index)

    def right_onPressed(event):
        global image_index, current_displayed_image
        if image_index < len(image_addresses):
            image_index += 1
            page_transit(image_index)

    def open_image(event):
        try:
            os.startfile(image_addresses[image_index-1][0])
        except (FileNotFoundError, IndexError) as e:
            pass

    def switch_to_sidebyside():
        global display_mode
        if display_mode != "sidebyside":
            display_mode = "sidebyside"
            base_canvas.pack(side="left", fill="both", before=image_canvas, expand=True)
            canvas_divider.pack(side="left", fill="y", pady=20, before=image_canvas)
            root.after(50, wait_until_base_canvas_ready_and_draw())
            root.after(50, func=lambda : wait_until_canvas_ready_and_draw("sidebyside"))

    def switch_to_single():
        global display_mode
        if display_mode != "single":
            display_mode = "single"
            base_canvas.pack_forget()
            canvas_divider.pack_forget()
            root.after(50, func=lambda : wait_until_canvas_ready_and_draw("single"))

    def wait_until_canvas_ready():
        global image_canvas_width, image_canvas_height
        canvas_w = image_canvas.winfo_width()
        canvas_h = image_canvas.winfo_height()
        if canvas_w > 10 and canvas_h > 10:
            image_canvas_width = canvas_w
            image_canvas_height = canvas_h
        else:
            root.after(50, wait_until_canvas_ready)

    def wait_until_canvas_ready_and_draw(mode):
        global current_displayed_image, image_canvas_width, image_canvas_height, found_photo
        canvas_w = image_canvas.winfo_width()
        canvas_h = image_canvas.winfo_height()
        if mode == "sidebyside":
            if canvas_w < image_canvas_width:
                image_canvas_width = canvas_w
                image_canvas_height = canvas_h
                found_image = None
                if len(image_addresses) > 0:
                    try:
                        try:
                            with Image.open(image_addresses[image_index-1][0]) as img:
                                found_image = img.copy()
                        except UnidentifiedImageError:
                            found_image = pdfium_wrapper.convert_page_to_image(image_addresses[image_index-1][0], image_addresses[image_index-1][1], 0, 0, 80)
                    except (OSError, FileNotFoundError, ValueError, Exception) as e:
                        found_image = not_found_img
                        print("Can't open file")
                    found_photo = ImageTk.PhotoImage(resize_image(found_image, image_canvas))
                    if current_displayed_image == 0:
                        current_displayed_image = image_canvas.create_image(image_canvas_width/2, image_canvas_height/2, anchor = CENTER, image=found_photo)
                    else:
                        image_canvas.delete(current_displayed_image)
                        current_displayed_image = 0
                        current_displayed_image = image_canvas.create_image(image_canvas_width/2, image_canvas_height/2, anchor = CENTER, image=found_photo)
            else:
                wait_until_canvas_ready_and_draw("sidebyside")
        if mode == "single":
            if canvas_w > image_canvas_width:
                image_canvas_width = canvas_w
                image_canvas_height = canvas_h
                found_image = None
                if len(image_addresses) > 0:
                    try:
                        try:
                            with Image.open(image_addresses[image_index-1][0]) as img:
                                found_image = img.copy()
                        except UnidentifiedImageError:
                            found_image = pdfium_wrapper.convert_page_to_image(image_addresses[image_index-1][0], image_addresses[image_index-1][1], 0, 0, 80)
                    except (OSError, FileNotFoundError, ValueError, Exception) as e:
                        found_image = not_found_img
                        print("Can't open file")
                    found_photo = ImageTk.PhotoImage(resize_image(found_image, image_canvas))
                    if current_displayed_image == 0:
                        current_displayed_image = image_canvas.create_image(image_canvas_width/2, image_canvas_height/2, anchor = CENTER, image=found_photo)
                    else:
                        image_canvas.delete(current_displayed_image)
                        current_displayed_image = 0
                        current_displayed_image = image_canvas.create_image(image_canvas_width/2, image_canvas_height/2, anchor = CENTER, image=found_photo)
            else:
                wait_until_canvas_ready_and_draw("single")

    def wait_until_base_canvas_ready_and_draw():
        global base_canvas_width, base_canvas_height, current_base_image, base_photo
        canvas_w = base_canvas.winfo_width()
        canvas_h = base_canvas.winfo_height()
        if canvas_w > 10 and canvas_h > 10:
            if base_address or (base_address == "" and base_image):
                base_canvas_width = canvas_w
                base_canvas_height = canvas_h
                base_photo = ImageTk.PhotoImage(resize_image(base_image, base_canvas))
                if current_base_image == 0:
                    current_base_image = base_canvas.create_image(base_canvas_width/2, base_canvas_height/2, anchor = CENTER, image=base_photo)
                else:
                    base_canvas.itemconfig(current_base_image, image = base_photo)
        else:
            root.after(50, wait_until_base_canvas_ready_and_draw)

    def take_screenshot():
        ScreenCapture()

    def onClosing():
        root.destroy()

    # GUI Widgets
    #Frames declaration
    topbar = Frame(home_tab, bg="white")
    content = Frame(home_tab)
    sidebar = ResizableFrame(content, width=250, bg="white", borderwidth=1, relief="ridge")
    main_area = Frame(content, bg="#F0F0F0")
    sidebar_content_frame = VerticalScrolledFrame(parent=sidebar)

    # ----- Top bar -----
    topbar.pack(side="top", fill="x", padx=10)

    # ----- Main content wrapper -----
    content.pack(side="top", fill="both", expand=True, padx=8, pady=8)

    # ----- Sidebar -----
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    # ----- Main image display area -----
    main_area.pack(side="left", fill="both", expand=True)

    #Sub-Frames
    left_frame = ttk.Frame(topbar)
    # end_frame = ttk.Frame(topbar)

    # Sub-Widgets
    base_address_label = ttk.Label(master=left_frame, text="Find  ", background="white")
    folder_address_label = ttk.Label(master=left_frame, text="Directory  ", background="white")

    base_address_value = StringVar(value="")
    folder_address_value = StringVar(value="")

    base_address_field = ttk.Entry(master=left_frame, width=170, textvariable=base_address_value, name="base_address_field")
    folder_address_field = ttk.Entry(master=left_frame, width=170, textvariable=folder_address_value, name="folder_address_field")

    base_address_browse = ttk.Button(master=left_frame, text="Browse", name="base_address_browse")
    folder_address_browse = ttk.Button(master=left_frame, text="Browse", name="folder_address_browse")

    ok_button = ttk.Button(master=left_frame, text="Search", image=search_icon, compound="right")

    screenshot_button = ttk.Button(master=left_frame, image=screenshot_icon, command=take_screenshot)

    main_bot_frame = Frame(master=main_area)
    score_frame = Frame(master=main_bot_frame)
    score_label = ttk.Label(master=score_frame)
    image_change_frame = Frame(master=main_bot_frame)
    left_button = ttk.Button(master= image_change_frame, image=left_icon)
    right_button = ttk.Button(master= image_change_frame, image=right_icon)

    main_mid_frame = Frame(master=main_area)
    image_canvas = Canvas(master=main_mid_frame)
    base_canvas = Canvas(master=main_mid_frame)
    canvas_divider = ttk.Separator(main_mid_frame, orient="vertical")
    image_counter_label = ttk.Label(master=image_change_frame, text="  0 / 0  ", )
    main_head_frame = Frame(master=main_area)
    image_address_var = StringVar()
    image_address_entry = ttk.Entry(master=main_head_frame, width=60, textvariable=image_address_var, justify="center", state="readonly")
    mode_frame = Frame(master=main_head_frame)
    single_mode_button = ttk.Button(master=mode_frame, image=single_icon, command=switch_to_single)
    sidebyside_mode_button = ttk.Button(master=mode_frame, image=sidebyside_icon, command=switch_to_sidebyside)

    property_label = ttk.Label(master=sidebar, text="Property:", background="white")

    change_meta_menu = Menu(master=home_tab, tearoff=0)
    change_meta_menu.add_command(label="Delete Property")


    # Layout
    left_frame.pack(side="top", fill="y", pady=8, padx=20)

    property_label.pack(side="top")

    sidebar_content_frame.pack(side="top", fill="both", expand=True, padx=20)
    sidebar_content_frame.pack_propagate(False)

    base_address_label.grid(row=0, column=0, sticky="w")
    base_address_field.grid(row=0, column=1)
    base_address_browse.grid(row=0, column=2)

    folder_address_label.grid(row=1, column=0, sticky="w")
    folder_address_field.grid(row=1, column=1)
    folder_address_browse.grid(row=1, column=2)

    ok_button.grid(row=3, column=1)

    screenshot_button.grid(row=0, column=3)

    main_head_frame.pack(side="top", fill="x")
    image_address_entry.pack(side="top", fill="x")
    mode_frame.pack(side="top", fill="x")
    single_mode_button.grid(row=0, column=0)
    sidebyside_mode_button.grid(row=0, column=1)

    main_mid_frame.pack(side="top", fill="both", expand=True)
    image_canvas.pack(side="left", fill="both", expand=True)

    main_bot_frame.pack(side="top")
    score_frame.pack(side="top")
    score_label.pack(side="top")
    image_change_frame.pack(side="top")
    left_button.grid(row=0, column=0)
    image_counter_label.grid(row=0, column=1)
    right_button.grid(row=0, column=2)
    # Events
    root.bind_all("<Control-z>", revert_onPressed)
    root.bind_all("<Command-z>", revert_onPressed)
    base_address_browse.bind("<Button-1>", base_onPressed)
    folder_address_browse.bind("<Button-1>", folder_onPressed)
    ok_button.bind("<Button-1>", ok_onPressed)
    image_canvas.bind("<Double-Button-1>", open_image)
    left_button.bind("<Button-1>", left_onPressed)
    right_button.bind("<Button-1>", right_onPressed)

    ####################

    ##SETTINGS TAB
    def is_wd_registered_startup():
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
                val, _ = winreg.QueryValueEx(key, WATCHDOG_NAME)
                print(val)
                return val == WATCHDOG_PATH
        except FileNotFoundError:
            return False
        
    def is_wd_running(script_name="WatchdogService.exe"):
        current_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if proc.info["pid"] == current_pid:
                    continue  # skip self
                cmdline = proc.info["cmdline"]
                if cmdline and script_name in " ".join(cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
        
    def add_wd_to_startup():
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, WATCHDOG_NAME, 0, winreg.REG_SZ, WATCHDOG_PATH)
        except PermissionError:
            print("Permission to add Watchdog is not provided")
            return

    def remove_wd_from_startup():
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE) as key:
            try:
                winreg.DeleteValue(key, WATCHDOG_NAME)
            except FileNotFoundError:
                pass

    def start_watchdog():
        pythonw = os.path.join(os.path.dirname(sys.executable), "python.exe")
        subprocess.Popen([pythonw, WATCHDOG_PATH])

    def run_wd_onPressed():
        if is_wd_running():
            print("running")
            return
        else:
            start_watchdog()

    def save_settings_onPressed():
        global image_accuracy
        #Watchdog
        boot_watchdog = boot_watchdog_var.get()
        if is_wd_registered_startup():
            if not boot_watchdog:
                remove_wd_from_startup()
        else:
            if boot_watchdog:
                add_wd_to_startup()

    setting_frame = VerticalScrolledFrame(parent=setting_tab)
    default_setting_frame = ttk.Frame(setting_frame.interior)
    boot_watchdog_frame = ttk.Frame(default_setting_frame)
    boot_watchdog_label = ttk.Label(boot_watchdog_frame, text="Boot watchdog with Windows")
    boot_watchdog_var = IntVar()
    boot_watchdog_check = ttk.Checkbutton(boot_watchdog_frame, variable=boot_watchdog_var)
    save_setting_button = ttk.Button(setting_tab, text="Save", command=save_settings_onPressed)

    run_wd_frame = ttk.Frame(default_setting_frame)
    run_wd_button = ttk.Button(run_wd_frame, text="Run Watchdog Service", command=run_wd_onPressed)

    accuracy_frame = ttk.Frame(master=default_setting_frame)
    accuracy_label = ttk.Label(master=accuracy_frame, text="Image Accuracy : ")
    accuracy_slider = ttk.Scale(master=accuracy_frame, orient="horizontal", from_=0, to=100)

    setting_frame.pack(expand=True, fill="both")

    default_setting_frame.pack(side="top", expand=True, fill="both", pady=40)
    boot_watchdog_frame.pack(side="top", pady=30)
    boot_watchdog_label.pack(side="left", padx=30)
    boot_watchdog_check.pack(side="left")

    run_wd_frame.pack(side="top", pady=30)
    run_wd_button.pack(side="top")

    accuracy_frame.pack(side="top", pady=30)
    accuracy_label.pack(side="left", padx=30)
    accuracy_slider.pack(side="left")

    save_setting_button.pack(side="top")

    if is_wd_registered_startup():
        boot_watchdog_var.set(1)
    else:
        boot_watchdog_var.set(0)

    root.protocol("WM_DELETE_WINDOW", onClosing)

    root.after(50, wait_until_canvas_ready)
    root.mainloop()

if __name__ == "__main__":
    if not pyuac.isUserAdmin():
        pyuac.runAsAdmin()
    load_map_data()
    load_worker_threads(3)
    load_MobileNetV3_small()
    run_gui()


