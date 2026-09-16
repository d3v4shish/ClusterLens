from __future__ import annotations

"""Consistent, keyboard-safe autocomplete controls for named local entities.

The widget deliberately owns only presentation and selection semantics.  Its
owner can populate choices from any cancellable background query without
coupling the Qt control to a database or service implementation.
"""

from collections.abc import Iterable

from PyQt6.QtCore import QEvent, QStringListModel, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QCompleter, QDialog, QDialogButtonBox, QLabel, QLineEdit, QVBoxLayout


class EntityPicker(QLineEdit):
    """A shared autocomplete input with explicit creation and commit rules.

    A highlighted saved value is committed by mouse, Enter, or Tab.  New
    values are never mistaken for a completion: the popup shows a distinct
    ``Create …`` row only when the caller permits creation.
    """

    value_committed = pyqtSignal(str)
    create_requested = pyqtSignal(str)

    _CREATE_PREFIX = "Create \u201c"
    _CREATE_SUFFIX = "\u201d"

    def __init__(self, parent=None, *, allow_create: bool = True, entity_label: str = "name"):
        super().__init__(parent)
        self._allow_create = bool(allow_create)
        self._entity_label = str(entity_label or "name")
        self._choices: list[str] = []
        self._choices_by_key: dict[str, str] = {}
        self._search_choices: list[tuple[str, str]] = []
        self._displayed_choices: list[str] = []
        self._show_completions_on_focus = True
        self._name_model = QStringListModel(self)
        self._completer = QCompleter(self._name_model, self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.activated[str].connect(self._commit_completion)
        self.setCompleter(self._completer)
        self.textEdited.connect(self._update_suggestions)
        self.setClearButtonEnabled(True)

    @property
    def choices(self) -> tuple[str, ...]:
        return tuple(self._choices)

    @property
    def _name_completer(self) -> QCompleter:
        """Compatibility alias for the previous Faces-only name editor."""
        return self._completer

    def set_entity_label(self, label: str) -> None:
        self._entity_label = str(label or "name")
        self._update_suggestions(self.text())

    def set_allow_create(self, enabled: bool) -> None:
        self._allow_create = bool(enabled)
        self._update_suggestions(self.text())

    def set_choices(self, values: Iterable[str] | None) -> None:
        unique: dict[str, str] = {}
        for raw_value in values or ():
            value = str(raw_value or "").strip()
            if value:
                unique.setdefault(value.casefold(), value)
        self._choices_by_key = dict(unique)
        self._search_choices = sorted(unique.items(), key=lambda item: (item[0], item[1]))
        self._choices = [value for _key, value in self._search_choices]
        self._update_suggestions(self.text())

    # Retain the existing Faces public method while making the control reusable.
    set_name_choices = set_choices

    def canonical_value(self, value: str | None = None) -> str:
        entered = str(self.text() if value is None else value or "").strip()
        if not entered:
            return ""
        return self._choices_by_key.get(entered.casefold(), entered)

    def _update_suggestions(self, text: str = "") -> None:
        entered = str(text or "").strip()
        folded = entered.casefold()
        matches = [choice for key, choice in self._search_choices if not folded or folded in key]
        if self._allow_create and entered and folded not in self._choices_by_key:
            matches.append(self._create_row(entered))
        self._displayed_choices = matches
        self._name_model.setStringList(matches)
        if self.hasFocus() and matches:
            QTimer.singleShot(0, self._completer.complete)

    @classmethod
    def _create_row(cls, value: str) -> str:
        return f"{cls._CREATE_PREFIX}{str(value).strip()}{cls._CREATE_SUFFIX}"

    @classmethod
    def _is_create_row(cls, value: str) -> bool:
        return str(value).startswith(cls._CREATE_PREFIX) and str(value).endswith(cls._CREATE_SUFFIX)

    @classmethod
    def _create_value(cls, value: str) -> str:
        text = str(value)
        if cls._is_create_row(text):
            return text[len(cls._CREATE_PREFIX) : -len(cls._CREATE_SUFFIX)].strip()
        return ""

    def _commit_completion(self, completion: str = "") -> bool:
        selected = str(completion or self._completer.currentCompletion() or "").strip()
        if not selected:
            return False
        if self._is_create_row(selected):
            created = self._create_value(selected)
            if not created:
                return False
            self.setText(created)
            self.setCursorPosition(len(created))
            self.create_requested.emit(created)
            self.value_committed.emit(created)
        else:
            canonical = self.canonical_value(selected)
            self.setText(canonical)
            self.setCursorPosition(len(canonical))
            self.value_committed.emit(canonical)
        self._show_completions_on_focus = False
        self._completer.popup().hide()
        return True

    def event(self, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.KeyPress and event.key() in {
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Tab,
        }:
            popup = self._completer.popup()
            # Some compositors cannot expose a completion popup in an
            # offscreen/embedded window even though QCompleter has a current
            # highlighted value. A non-empty prefix is the same explicit user
            # intent as that visible popup, so preserve Enter/Tab semantics.
            if (popup.isVisible() or self._completer.completionPrefix()) and self._commit_completion():
                event.accept()
                return True
        return super().event(event)

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        if self._show_completions_on_focus and self._choices:
            self._update_suggestions(self.text())


class EntityPickerDialog(QDialog):
    """Small reusable picker dialog for a name assignment or rename."""

    def __init__(
        self,
        title: str,
        label: str,
        *,
        choices: Iterable[str] | None = None,
        initial: str = "",
        allow_create: bool = True,
        entity_label: str = "name",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Choose value"))
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        prompt = QLabel(str(label or "Choose a value"), self)
        prompt.setWordWrap(True)
        layout.addWidget(prompt)
        self.value_input = EntityPicker(self, allow_create=allow_create, entity_label=entity_label)
        self.value_input.set_choices(choices)
        self.value_input.setText(str(initial or ""))
        self.value_input.selectAll()
        self.value_input.setPlaceholderText(
            f"Type a new {entity_label} or choose a saved {entity_label}"
            if allow_create
            else f"Choose a saved {entity_label}"
        )
        layout.addWidget(self.value_input)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(420, self.sizeHint().height())

    def selected_value(self) -> str:
        return self.value_input.canonical_value()
