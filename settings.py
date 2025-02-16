import os
import yaml
import logging
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog
from PyQt5.uic import loadUi


class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super(SettingsWindow, self).__init__(parent)
        loadUi("resources/settings.ui", self)
        
        # Example: Initialize settings fields with existing values from MainWindow
        self.apiTokenInput.setText(parent.api_token)
        self.embeddingModelInput.setText(parent.embedding_model)
        self.llmModelInput.setText(parent.llm_model)
        self.chunkSizeInput.setValue(parent.chunk_size)
        self.chunkOverlapInput.setValue(parent.chunk_overlap)
        self.interfaceModeDropdown.setCurrentText(parent.interface_mode)

        # Example: Directory fields (optional)
        self.documentsDirInput.setText(parent.internal_folder)
        self.chatHistoryDirInput.setText(parent.chat_history_dir)
        
        # Save Button Action
        self.saveButton.clicked.connect(self.save_settings)
        self.cancelButton.clicked.connect(self.close)
        self.resetButton.clicked.connect(self.reset_settings)
        self.browseDocumentsButton.clicked.connect(self.browse_dir)
        self.browseChatHistoryButton.clicked.connect(self.browse_dir)

        self.parent = parent
    
    def reset_settings(self):
        self.apiTokenInput.setText(self.parent.api_token)
        self.embeddingModelInput.setText(self.parent.embedding_model)
        self.llmModelInput.setText(self.parent.llm_model)
        self.chunkSizeInput.setValue(self.parent.chunk_size)
        self.chunkOverlapInput.setValue(self.parent.chunk_overlap)
        self.interfaceModeDropdown.setCurrentText(self.parent.interface_mode)
        self.documentsDirInput.setText(self.parent.internal_folder)
        self.chatHistoryDirInput.setText(self.parent.chat_history_dir)

    def save_settings(self):
        """Save the settings and update MainWindow values."""
        self.parent.api_token = self.apiTokenInput.text()
        self.parent.embedding_model = self.embeddingModelInput.text()
        self.parent.llm_model = self.llmModelInput.text()
        self.parent.chunk_size = self.chunkSizeInput.value()
        self.parent.chunk_overlap = self.chunkOverlapInput.value()
        self.parent.interface_mode = self.interfaceModeDropdown.currentText().upper()
        self.parent.internal_folder = self.documentsDirInput.text()
        self.parent.chat_history_dir = self.chatHistoryDirInput.text()

        # Save changes to config.yaml
        self.parent.config.update({
            "API_TOKEN": self.parent.api_token,
            "EMBEDDING_MODEL": self.parent.embedding_model,
            "LLM_MODEL": self.parent.llm_model,
            "CHUNK_SIZE": self.parent.chunk_size,
            "CHUNK_OVERLAP": self.parent.chunk_overlap,
            "INTERFACE_MODE": self.parent.interface_mode,
            "DOC_DIR": self.parent.internal_folder,
            "CHAT_DIR": self.parent.chat_history_dir
        })
        with open("config.yaml", "w") as config_file:
            yaml.dump(self.parent.config, config_file)

        logging.info("Settings saved and applied.")
        QMessageBox.information(self, "Settings Saved", "The effect will be applied after restarting the application.")
        self.close()

    def browse_dir(self):
        """Open a file dialog to select a directory."""
        directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if directory:
            if self.sender() == self.browseDocumentsButton:
                self.documentsDirInput.setText(directory)
            elif self.sender() == self.browseChatHistoryButton:
                self.chatHistoryDirInput.setText(directory)