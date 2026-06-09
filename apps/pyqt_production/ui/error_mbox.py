from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def errorBox(msg_title: str, msg_info: str = "") -> None:
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText(msg_title)
    msg.setInformativeText(msg_info)
    msg.setWindowTitle("Error")
    msg.exec()


def infoBox(msg_title: str, msg_info: str = "") -> None:
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setText(msg_title)
    msg.setInformativeText(msg_info)
    msg.setWindowTitle("Info")
    msg.exec()


def confirmBox(msg_title: str, msg_info: str = "", *, parent=None) -> bool:
    result = QMessageBox.question(
        parent,
        msg_title,
        msg_info,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return result == QMessageBox.StandardButton.Yes


class InputMessageBox(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input")
        layout = QVBoxLayout(self)
        self.label = QLabel("Enter your text:")
        layout.addWidget(self.label)
        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def getText(self) -> str:
        return self.input_field.text().strip()


class ExifMetadataDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Write EXIF Metadata")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self.key_field = QLineEdit(self)
        self.value_field = QLineEdit(self)
        self.preview_label = QLabel("Preview: ")
        self.preview_label.setWordWrap(True)

        form.addRow("Key", self.key_field)
        form.addRow("Value", self.value_field)
        layout.addLayout(form)
        layout.addWidget(self.preview_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.key_field.textChanged.connect(self._update_preview)
        self.value_field.textChanged.connect(self._update_preview)
        self._update_preview()

    def _update_preview(self) -> None:
        key = self.key_field.text().strip()
        value = self.value_field.text().strip()
        preview = f"{key}: {value}".strip(": ").strip()
        self.preview_label.setText(f"Preview: {preview}")

    def get_pair(self) -> tuple[str, str]:
        return self.key_field.text().strip(), self.value_field.text().strip()


class ImageTagsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Image Tags")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self.mode_combobox = QComboBox(self)
        self.mode_combobox.addItems(["Add", "Remove"])
        self.tags_field = QLineEdit(self)
        self.tags_field.setPlaceholderText("Comma-separated tags")
        self.mirror_checkbox = QCheckBox("Mirror tags to EXIF UserComment", self)
        self.mirror_checkbox.setChecked(True)
        self.preview_label = QLabel("Preview: Add []")
        self.preview_label.setWordWrap(True)

        form.addRow("Mode", self.mode_combobox)
        form.addRow("Tags", self.tags_field)
        layout.addLayout(form)
        layout.addWidget(self.mirror_checkbox)
        layout.addWidget(self.preview_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.mode_combobox.currentIndexChanged.connect(self._update_preview)
        self.tags_field.textChanged.connect(self._update_preview)
        self._update_preview()

    def _update_preview(self) -> None:
        mode = self.mode_combobox.currentText().strip() or "Add"
        tags = [tag.strip() for tag in self.tags_field.text().split(",") if tag.strip()]
        self.preview_label.setText(f"Preview: {mode} {tags}")

    def values(self) -> tuple[str, str, bool]:
        return (
            self.mode_combobox.currentText().strip() or "Add",
            self.tags_field.text().strip(),
            bool(self.mirror_checkbox.isChecked()),
        )


class TagManagerDialog(QDialog):
    def __init__(self, tag_service, parent=None):
        super().__init__(parent)
        self.tag_service = tag_service
        self.setWindowTitle("Tag Manager")
        self.resize(720, 520)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Manage app tag database entries. Rename merges into an existing tag when needed. "
            "These operations update the app tag database, not EXIF sidecar data."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Tag", "Images", "Sources", "Normalized"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._sync_selected_tag)
        layout.addWidget(self.table, stretch=1)

        form = QFormLayout()
        self.selected_tag_field = QLineEdit(self)
        self.selected_tag_field.setPlaceholderText("Selected tag")
        self.rename_field = QLineEdit(self)
        self.rename_field.setPlaceholderText("New tag name")
        form.addRow("Selected", self.selected_tag_field)
        form.addRow("Rename / Merge To", self.rename_field)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh", self)
        self.rename_button = QPushButton("Rename / Merge", self)
        self.delete_button = QPushButton("Delete Tag", self)
        actions.addWidget(self.refresh_button)
        actions.addStretch(1)
        actions.addWidget(self.rename_button)
        actions.addWidget(self.delete_button)
        layout.addLayout(actions)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        layout.addWidget(self.button_box)
        self.button_box.rejected.connect(self.reject)
        self.refresh_button.clicked.connect(self.refresh)
        self.rename_button.clicked.connect(self._rename_selected)
        self.delete_button.clicked.connect(self._delete_selected)
        self.refresh()

    def refresh(self) -> None:
        inventory = self.tag_service.list_tag_inventory()
        self.table.setRowCount(len(inventory))
        for row, item in enumerate(inventory):
            sources = ", ".join(f"{source} ({count})" for source, count in item.sources) or "unknown"
            values = [
                item.display_tag,
                str(item.image_count),
                sources,
                item.normalized_tag,
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column == 0:
                    table_item.setData(256, item.display_tag)
                self.table.setItem(row, column, table_item)
        self.table.resizeColumnsToContents()
        self._sync_selected_tag()

    def selected_tag(self) -> str:
        text = self.selected_tag_field.text().strip()
        if text:
            return text
        selected = self.table.selectedItems()
        if not selected:
            return ""
        return self.table.item(selected[0].row(), 0).text().strip()

    def _sync_selected_tag(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            return
        tag = self.table.item(selected[0].row(), 0).text().strip()
        self.selected_tag_field.setText(tag)
        self.rename_field.setText(tag)

    def _rename_selected(self) -> None:
        old_tag = self.selected_tag()
        new_tag = self.rename_field.text().strip()
        if not old_tag or not new_tag:
            infoBox("Rename unavailable", "Select a tag and enter a replacement name.")
            return
        affected = self.tag_service.rename_tag(old_tag, new_tag)
        infoBox("Tag renamed", f"Updated {affected} image(s).")
        self.refresh()

    def _delete_selected(self) -> None:
        tag = self.selected_tag()
        if not tag:
            infoBox("Delete unavailable", "Select a tag first.")
            return
        if not confirmBox("Delete tag", f"Remove tag '{tag}' from all images?", parent=self):
            return
        affected = self.tag_service.delete_tag(tag)
        infoBox("Tag deleted", f"Removed from {affected} image(s).")
        self.refresh()
