from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COORDINATED_ASYNC_PRODUCERS = (
    "apps/pyqt_production/app.py",
    "apps/pyqt_production/settings_dialog.py",
    "src/ui/gallery_pane.py",
    "src/ui/sectioned_gallery.py",
    "src/ui/library_pane.py",
    "src/ui/tags_pane.py",
    "src/ui/names_pane.py",
    "src/ui/photo_inspector_dialog.py",
    "src/ui/cluster_pane.py",
    "src/ui/search_pane.py",
)

BOUNDED_INTERNAL_THREAD_STARTS = {
    ("src/ui/gallery_pane.py", "_ensure_loader"),
    ("src/ui/search_pane.py", "_start_face_tile_loader_threads"),
}

WORKER_FUNCTION_NAMES = {"_run", "_write", "_read_preview"}
QWIDGET_GETTERS = {
    "checkState",
    "currentData",
    "currentIndex",
    "currentItem",
    "currentText",
    "isChecked",
    "isEnabled",
    "isVisible",
    "selectedIndexes",
    "selectedItems",
    "text",
    "toPlainText",
    "value",
}
QWIDGET_MUTATORS = {
    "addItem",
    "addItems",
    "clear",
    "hide",
    "setChecked",
    "setCurrentIndex",
    "setCurrentText",
    "setEnabled",
    "setHtml",
    "setModel",
    "setPixmap",
    "setPlainText",
    "setRange",
    "setText",
    "setValue",
    "setVisible",
    "show",
}
LIVE_UI_HELPERS = {
    "_active_face_service",
    "_active_search_service",
    "_current_directory",
    "_current_scope_paths",
    "_current_scope_roots",
    "_show_tiny_detections_enabled",
    "_use_onnx",
    "current_face_detector_id",
    "current_face_detector_score_threshold",
    "current_face_embedder_id",
    "current_face_max_detections",
    "current_face_mode",
}


class _CallInventory(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: list[str] = []
        self.async_starts: list[tuple[int, str]] = []
        self.thread_starts: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        owner = self.functions[-1] if self.functions else "<module>"
        if isinstance(node.func, ast.Name) and node.func.id == "start_job_in_thread":
            self.async_starts.append((node.lineno, owner))
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "start"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "thread"
        ):
            self.thread_starts.append((node.lineno, owner))
        self.generic_visit(node)


def _inventory(relative_path: str) -> _CallInventory:
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    visitor = _CallInventory()
    visitor.visit(ast.parse(source, filename=relative_path))
    return visitor


def test_every_production_async_job_starts_only_inside_an_admitted_launcher() -> None:
    violations: list[str] = []
    for relative_path in COORDINATED_ASYNC_PRODUCERS:
        for lineno, owner in _inventory(relative_path).async_starts:
            if owner != "_launch":
                violations.append(f"{relative_path}:{lineno} starts in {owner}")
    assert violations == []


def test_direct_ui_thread_starts_are_only_bounded_thumbnail_pools() -> None:
    violations: list[str] = []
    for relative_path in COORDINATED_ASYNC_PRODUCERS:
        for lineno, owner in _inventory(relative_path).thread_starts:
            # AsyncJob launchers call the shared helper rather than QThread.start.
            # These two long-lived pools consume bounded queues, publish only
            # generation-checked thumbnails, and never mutate durable data.
            if (relative_path, owner) not in BOUNDED_INTERNAL_THREAD_STARTS:
                violations.append(f"{relative_path}:{lineno} starts in {owner}")
    assert violations == []


def test_process_controllers_are_only_started_by_coordinator_callbacks() -> None:
    source = (ROOT / "apps/pyqt_production/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    parent_stack: list[str] = []
    starts: list[tuple[int, str, str]] = []

    class _ControllerStarts(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            parent_stack.append(node.name)
            self.generic_visit(node)
            parent_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "start"
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
                and node.func.value.attr in {"session_controller", "model_download_controller"}
            ):
                starts.append((node.lineno, parent_stack[-1], node.func.value.attr))
            self.generic_visit(node)

    _ControllerStarts().visit(tree)
    assert [(owner, name) for _line, owner, name in starts] == [
        ("_start", "session_controller"),
        ("_start", "model_download_controller"),
    ]


def test_worker_closures_do_not_read_live_widgets_or_ui_context_helpers() -> None:
    """Production workers must consume immutable values captured on Qt."""

    violations: list[str] = []
    for relative_path in COORDINATED_ASYNC_PRODUCERS:
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"), filename=relative_path)
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if function.name not in WORKER_FUNCTION_NAMES:
                continue
            for call in ast.walk(function):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    continue
                target = call.func
                if (
                    isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in LIVE_UI_HELPERS
                ):
                    violations.append(
                        f"{relative_path}:{call.lineno} worker calls live helper self.{target.attr}()"
                    )
                if target.attr not in QWIDGET_GETTERS | QWIDGET_MUTATORS:
                    continue
                root = target.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id == "self":
                    violations.append(
                        f"{relative_path}:{call.lineno} worker accesses QWidget.{target.attr}()"
                    )
    assert violations == []


def test_qt_launch_paths_do_not_probe_or_construct_model_providers() -> None:
    source = (ROOT / "apps/pyqt_production/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls: list[str] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if function.name not in {"_start_request", "_apply_runtime_status"}:
            continue
        for call in ast.walk(function):
            if not isinstance(call, ast.Call):
                continue
            rendered = ast.unparse(call.func)
            if rendered in {
                "self.runtime_service.detect",
                "self.runtime_service.select_policy",
            } or rendered.endswith(("InferenceSession", "ModelManager")):
                forbidden_calls.append(
                    f"apps/pyqt_production/app.py:{call.lineno} {function.name} calls {rendered}"
                )
    assert forbidden_calls == []


def test_identity_and_saved_search_click_handlers_do_not_perform_storage_io() -> None:
    source = (ROOT / "src/ui/search_pane.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    handlers = {
        "_clear_selected_identity_labels",
        "_delete_selected_saved_search",
        "_merge_selected_identity_into_target",
        "_pin_selected_identity_prototype_face",
        "_remove_selected_identity_prototype_faces",
        "_rename_selected_saved_search",
        "_save_current_saved_search",
        "_selected_saved_search",
    }
    storage_methods = {
        "clear_person_labels",
        "delete_search",
        "get_search",
        "identity_duplicate_warnings",
        "label_counts",
        "list_searches",
        "merge_person_identities",
        "pin_person_prototype_face",
        "remove_person_prototype_face",
        "rename_search",
        "save_search",
    }
    violations: list[str] = []

    class _QtHandlerCalls(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Worker closures are the intended storage boundary.
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr in storage_methods:
                violations.append(
                    f"src/ui/search_pane.py:{node.lineno} Qt handler calls {node.func.attr}()"
                )
            self.generic_visit(node)

    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name not in handlers:
            continue
        visitor = _QtHandlerCalls()
        for statement in function.body:
            visitor.visit(statement)
    assert violations == []


def test_library_navigation_storage_reads_are_confined_to_worker_closures() -> None:
    relative_path = "src/ui/library_pane.py"
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"), filename=relative_path)
    storage_reads = {"get_smart_album", "list_roots", "list_smart_albums", "root_asset_counts"}
    violations: list[str] = []
    function_stack: list[str] = []

    class _CatalogReads(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in storage_reads
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
                and node.func.value.attr == "catalog"
                and (not function_stack or function_stack[-1] != "_run")
            ):
                violations.append(
                    f"{relative_path}:{node.lineno} {function_stack[-1]} calls {node.func.attr}()"
                )
            self.generic_visit(node)

    _CatalogReads().visit(tree)
    assert violations == []


def test_frequent_health_and_settings_reload_paths_do_not_scan_or_construct_services() -> None:
    relative_path = "apps/pyqt_production/app.py"
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"), filename=relative_path)
    forbidden_by_function = {
        "_refresh_health_badge": {"bundled_model_names", "validate_bundle", "find_bundle"},
        "_reload_face_services_from_settings": {
            "_build_face_services",
            "_face_service_for_pipeline",
            "FaceIndexService",
        },
        "_face_service_for_pipeline": {"FaceIndexService", "resolve_ready_face_pipeline_ids"},
    }
    violations: list[str] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name not in forbidden_by_function:
            continue
        forbidden = forbidden_by_function[function.name]
        for call in ast.walk(function):
            if not isinstance(call, ast.Call):
                continue
            rendered = ast.unparse(call.func)
            if rendered.split(".")[-1] in forbidden:
                violations.append(f"{relative_path}:{call.lineno} {function.name} calls {rendered}")
    assert violations == []


def test_people_maintenance_and_paste_handlers_do_not_perform_storage_io() -> None:
    relative_path = "src/ui/search_pane.py"
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"), filename=relative_path)
    handlers = {"_purge_face_data", "_set_face_recognition_enabled", "_try_paste_image"}
    forbidden = {
        "mkdir",
        "open",
        "prepare_face_data_purge",
        "purge_face_data",
        "save",
        "set_face_recognition_enabled",
        "unlink",
    }
    violations: list[str] = []

    class _DirectHandlerCalls(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Nested worker closures are the intended storage boundary.
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
                violations.append(
                    f"{relative_path}:{node.lineno} Qt handler calls {ast.unparse(node.func)}"
                )
            self.generic_visit(node)

    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name not in handlers:
            continue
        visitor = _DirectHandlerCalls()
        for statement in function.body:
            visitor.visit(statement)
    assert violations == []


def test_gallery_result_partitioning_is_prepared_in_the_discovery_worker() -> None:
    relative_path = "apps/pyqt_production/app.py"
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"), filename=relative_path)
    target = next(
        function
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef) and function.name == "_load_gallery_scope"
    )
    nested = {
        function.name: function
        for function in ast.walk(target)
        if isinstance(function, ast.FunctionDef) and function is not target
    }
    assert "_run" in nested
    assert "_finished" in nested
    worker_calls = {ast.unparse(call.func) for call in ast.walk(nested["_run"]) if isinstance(call, ast.Call)}
    commit_calls = {ast.unparse(call.func) for call in ast.walk(nested["_finished"]) if isinstance(call, ast.Call)}
    assert "root_scope.contains" in worker_calls
    assert not any(call.endswith(("PathScope.from_paths", ".contains")) for call in commit_calls)


def test_source_restore_and_recent_history_construction_do_not_probe_filesystems() -> None:
    source_pane = (ROOT / "src/ui/source_pane.py").read_text(encoding="utf-8")
    recent_folders = (ROOT / "src/ui/recent_folders.py").read_text(encoding="utf-8")
    app_source = (ROOT / "apps/pyqt_production/app.py").read_text(encoding="utf-8")
    source_tree = ast.parse(source_pane, filename="src/ui/source_pane.py")
    recent_tree = ast.parse(recent_folders, filename="src/ui/recent_folders.py")
    app_tree = ast.parse(app_source, filename="apps/pyqt_production/app.py")

    def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
        return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)

    selected_calls = {
        ast.unparse(call.func)
        for call in ast.walk(_function(source_tree, "set_selected_directory"))
        if isinstance(call, ast.Call)
    }
    recent_init_calls = {
        ast.unparse(call.func)
        for call in ast.walk(_function(recent_tree, "__init__"))
        if isinstance(call, ast.Call)
    }
    restore_calls = {
        ast.unparse(call.func)
        for call in ast.walk(_function(app_tree, "_restore_active_roots"))
        if isinstance(call, ast.Call)
    }
    assert "os.path.exists" not in selected_calls
    assert "self._save" not in recent_init_calls
    assert "self._valid_directory" not in recent_init_calls
    assert "PathScope.from_paths" in restore_calls
    restore_call = next(
        call
        for call in ast.walk(_function(app_tree, "_restore_active_roots"))
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "PathScope.from_paths"
    )
    assert any(keyword.arg == "resolve_symlinks" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False for keyword in restore_call.keywords)
