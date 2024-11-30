import numpy as np
import cv2
import csv
from os import listdir, path
from app.BaseDetection import BaseDetection
from app.ImageViewer import ImageViewer
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, 
    QVBoxLayout, QLayout, QLabel,
    QTableView, QMessageBox,
    QPushButton, QFileDialog
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont, QColor

class MultipleDetection(BaseDetection):
    def __init__(self, show_page_callback):
        super().__init__(show_page_callback)
        
        # no-ui instances
        self.images_path = ""
        self.filenames = []
        self.image_viewer_refs = [] # handle references to image viewer
        
        # unique widgets  
        self.result = QLabel("Detection Results", alignment=Qt.AlignCenter)
        self.result.setProperty("class", "tumor_detection")
        self.csv_export = QPushButton("Export to CSV")
        self.show_btn = QPushButton("Show")
        
        # create table
        self.table_model = QStandardItemModel(0, 2)
        self.table = QTableView()
             
        # main layout of this window
        self.master_layout = QVBoxLayout()
        self.hrows_layout = [QHBoxLayout() for _ in range(3)]

        # connect buttons
        self.csv_export.clicked.connect(self.export_to_csv)
        self.show_btn.clicked.connect(self.show_image)        

        # setup interface
        self.setup_interface()

          
    def setup_interface(self):        
        # setup first row
        self.hrows_layout[0].addWidget(self.go_back)
        self.hrows_layout[0].addWidget(self.browser)
        self.hrows_layout[0].addWidget(self.path)
        self.hrows_layout[0].addWidget(self.model_info)
        self.hrows_layout[0].addWidget(self.upload_model)
        
        # setup second row
        self.hrows_layout[1].addWidget(self.table)
        
        # setup third row
        self.hrows_layout[2].addWidget(self.csv_export)
        self.hrows_layout[2].addWidget(self.show_btn)
        self.hrows_layout[2].addWidget(self.result)
        self.hrows_layout[2].addWidget(self.detect)
    
        # add model to the table    
        self.table.setModel(self.table_model)
        self.set_table_view()
    
        # configure widgets position
        self.table.setFixedSize(1000, 480)
        self.csv_export.setFixedSize(155, 50)
        self.show_btn.setFixedSize(155, 50)
                
        # margins
        self.hrows_layout[0].contentsMargins()
        
        # add rows to master
        for row in self.hrows_layout:
            self.master_layout.addLayout(row)
        
        # set master
        self.setLayout(self.master_layout)

    def set_table_view(self):
        self.table_model.setHorizontalHeaderLabels(["File Name", "Detection Result"])
        self.table.setColumnWidth(0, 600)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)   

    def browse_for_img(self):
        """ Open file dialog to point out the path to the folder
        """
        def filter_(files, extensions):
            "function takes path, extension and look for paths to specified images"
            results = []
            for file in files:
                for ext in extensions:
                    if file.endswith(ext):
                        results.append(file)
            return results
        
        self.images_path = QFileDialog.getExistingDirectory()
        extensions = ['.png', '.jpg', '.jpeg', '.svg', '.bmp']
        # list path names of every image inside the directory
        try:
            self.filenames = filter_(listdir(self.images_path), extensions)
        except FileNotFoundError:
            QMessageBox.warning(self, "No path provided", "No path provided")
            return None
        if not self.filenames:
            QMessageBox.warning(self, "No Images", "No images in directory.")
            return None            
        if self.images_path:
            # set detection results to default
            self.result.setText("Detection Results")
            # clear table
            self.table_model.clear()
            # insert path to the qlineedit
            self.path.setText(self.images_path)
            # insert filenames
            for image in self.filenames:
                name = QStandardItem(str(image))
                default_detection = QStandardItem(" ")
                self.table_model.appendRow([name, default_detection])
            self.set_table_view()
            self.style().polish(self.table)            
        else:
            QMessageBox.warning(self, "No Directory Chosen", "To load images, choose directory.")
            return None

    def detect_tumor(self):
        # counter for final result info
        num_of_detections = 0
        num_of_files = len(self.filenames)
        self.result.setText("Detecting...")
        for idx in range(num_of_files):
            curr_directory = path.join(self.images_path, self.filenames[idx])
            curr_image = cv2.imread(curr_directory)
            # preprocess loaded image
            prep_curr_image = self.preprocess_image(curr_image)
            # model classifies image
            output = self.model.predict(prep_curr_image)
            if np.argmax(output) == 1:
                num_of_detections += 1
                item = QStandardItem("DETECTED")
                # qstandarditem cannot be modified by css styling, so
                # it's needed to do it manually
                item.setForeground(QColor(255, 0, 0)) 
                
            else:
                item = QStandardItem("No Tumor")
                # qstandarditem cannot be modified by css styling, so
                # it's needed to do it manually
                item.setForeground(QColor(0, 187, 0)) 
            # common styling
            font = QFont("Courier")
            font.setPointSize(12)
            font.setWeight(QFont.Bold)
            item.setFont(font)
            # insert item to the column
            self.table_model.setItem(idx, 1, item)
        self.result.setText(f"Detected: {num_of_detections} / {num_of_files}")

    def export_to_csv(self):
        """Take data from table and save it into .csv file.
        """
        filepath, _ = QFileDialog.getSaveFileName(self, 
                                                  "Save file as...",
                                                  "", 
                                                  "CSV Files (*.csv)")
        if filepath:
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # get lengths of table
                rows = self.table_model.rowCount()
                cols = self.table_model.columnCount()
                # add headers
                headers = [self.table_model.headerData(col, Qt.Horizontal) for col in range(cols)]
                writer.writerow(headers)
                # iterate over the table
                for row in range(rows):
                    # image name
                    row_data = []
                    item_name = self.table_model.item(row, 0)
                    row_data.append(item_name.text() if item_name else "")
                    item_result = self.table_model.item(row, 1)
                    # possibilities created for optional words which user can input after personal evalutation 
                    possibilites_yes = ('detected', 'yes', '1', 'true', 'y', 't')
                    possibilities_no = ('no tumor', 'no', '0', 'false', 'n', 'f')
                    # manipulate string to ease the compare to possibilities
                    check_pattern = str(item_result.text()).strip().lower()
                    # check table cell and insert data to csv depends on the content
                    row_data.append(
                        1 if item_result and check_pattern in possibilites_yes else 
                        0 if item_result and check_pattern in possibilities_no else 
                        "N/A")                    
                    writer.writerow(row_data)
            QMessageBox.information(self, "Success", f"File saved in {filepath}")
        else:
            QMessageBox.warning(self, "No Directory Chosen", "To save as .csv, choose Directory.")
            return None
    
    def apply_grad_cam(self):
        pass
    
    def show_image(self):
        selected_indexes = self.table.selectedIndexes()
        # prevent displaying too much
        if len(selected_indexes) < 10:
            for idx in selected_indexes:
                num = idx.row()
                filename = self.filenames[num]
                full_path = path.join(self.images_path, filename)
                image_viewer = ImageViewer(full_path)
                self.image_viewer_refs.append(image_viewer)

        else:
            QMessageBox.information(self, "Too much cells", "Choose up to 10 cells.")
            
