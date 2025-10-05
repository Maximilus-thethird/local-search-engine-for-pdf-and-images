# local-search-engine-for-pdf-and-images (QLen)

This is a personal proof-of-concept project developed in the summer following my first year as a CS student. The application is a local search engine with a graphical UI that allows users to search across PDFs and image files, edit metadata, and manage large-scale file databases efficiently.

This project was developed as a personal challenge to explore file indexing, metadata management, and local search optimization. It reflects my growing interest in systems programming and practical applications of computer vision and document parsing.

This application is a fully local, cross-type search engine for PDFs and image files. Users can initiate searches using either file paths or the built-in screenshot capture system. The engine returns exact matches or contextually similar images — for example, submitting a photo of a dog will retrieve all dog-related images within the indexed database.

Currently, this version only supports Windows.

##  Key Features

- **Cross-Type Search**  
  Intelligent matching between PDFs and images using visual and textual cues. Queries can be made via file paths or live screenshots.

- **Contextual Image Retrieval**  
  Returns both exact matches and semantically similar results based on image content, enabling flexible and intuitive discovery.

- **Metadata Editing (PoC Phase)**  
  Users can manipulate PDF metadata via an XML tree editor:
  - Add/remove nodes or entire trees
  - Import/export XML structures
  - Robust undo system for safe experimentation  
  _Note: Final XML output is not yet RDF-compliant, so changes may not be visible to external applications. This is planned for future updates._

- **Live File Monitoring**  
  New files added to the database are automatically indexed and made searchable in real time.  
  _Note: Without `WatchdogService` invoked from `FindImages.py`, new additions require a restart of `FindImages.py` to be reflected if `FindImages.py` is open during the file addition process._

- **Startup Integration (WIP)**  
  A boot-with-Windows feature for the monitoring service is available but under development to support persistent background indexing.

---

##  Tech Stack

- Python 3.11  
- PyMuPDF (fitz)  
- TorchVision / MobileNetV3  
- PDFium  
- Tkinter (GUI)
- SQLite

# Setup Guide

To use the application:

1. **Index the Database**  
   Run `SetupIndex.py` and select the desired folder to index. This will create a searchable database of PDFs and image files.

2. **Run Queries**  
   Launch `FindImages.py` and choose your query input:
   - Select a file path to an image or PDF
   - Or use the built-in screenshot system to capture a query image  
   Then, select the path to the previously indexed database.

3. **Edit PDF Metadata**  
   For matched PDF results, users can edit custom metadata via an XML tree editor:
   - Add or remove nodes
   - Import/export XML trees
   - Undo changes safely  
   _Note: Metadata editing is in proof-of-concept phase. Final XML output is not yet RDF-compliant, so changes may not be visible to external applications._

4. **Enable Live File Monitoring**  
   To automatically index new files:
   - Go to the **Settings** tab in the GUI
   - Click **Run Watchdog Service**  
   The service will monitor the database folder and index new additions in real time.

5. **Stop the Watchdog Service**  
   To close the service:
   - Locate the hidden tray icon in the Windows taskbar
   - Right-click the icon and select **Exit**

---

# Third-Party Licenses

This project uses the following third-party libraries:

- **PDFium** — Licensed under a BSD-style license. See [`third_party_licenses/PDFium_LICENSE.txt`](third_party_licenses/PDFium_LICENSE.txt)
- **TorchVision / MobileNetV3** — Licensed under a BSD-style license. See [`third_party_licenses/TorchVision_LICENSE.txt`](third_party_licenses/TorchVision_LICENSE.txt)
- PyMuPdf (fitz) - Licensed under GNU AGPL v3.0. See [`third_party_licenses/PyMuPDF_LICENSE.txt`](third_party_licenses/PyMuPDF_LICENSE.txt)

These components are used in accordance with their respective licenses. This project does not redistribute modified versions of these libraries.

# License

This project is licensed under a custom license.  
See [`LICENSE`](LICENSE) for details.

However, this project includes third-party components licensed under GNU AGPL v3.0 (e.g., PyMuPDF).  
As a result, the combined work must comply with AGPL v3.0 if redistributed or deployed publicly.

If you intend to reuse or modify this project, please review the AGPL v3.0 terms carefully.  
