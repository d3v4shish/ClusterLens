from __future__ import annotations


PRODUCTION_APP_ID = "ClusterLens"
PRODUCTION_DISPLAY_NAME = "ClusterLens"
PRODUCTION_QSETTINGS_ORG = "ClusterLens"
PRODUCTION_QSETTINGS_APP = "ClusterLens"

LEGACY_PRODUCTION_APP_ID = "ImageClusteringPyQtProduction"
LEGACY_PRODUCTION_DISPLAY_NAME = "Image Clustering Production"
LEGACY_PRODUCTION_APP_IDS = (LEGACY_PRODUCTION_APP_ID,)
LEGACY_PRODUCTION_QSETTINGS = ((LEGACY_PRODUCTION_APP_ID, LEGACY_PRODUCTION_APP_ID),)

MODEL_ASSETS_APP_ID = "ClusterLensModelAssets"
LEGACY_MODEL_ASSETS_APP_ID = "ImageClusteringPyQtModelAssets"
LEGACY_BENCHMARK_APP_ID = "ClusterLensLegacyBenchmark"


def production_settings_store():
    from PyQt6.QtCore import QSettings

    target = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
    migration_key = "identity_migration/legacy_qsettings_copied"
    if target.value(migration_key, False, bool):
        return target

    copied = 0
    for org, app in LEGACY_PRODUCTION_QSETTINGS:
        legacy = QSettings(org, app)
        for key in legacy.allKeys():
            if target.contains(key):
                continue
            target.setValue(key, legacy.value(key))
            copied += 1

    target.setValue(migration_key, True)
    target.setValue("identity/app_id", PRODUCTION_APP_ID)
    target.setValue("identity/legacy_qsettings_copied_count", copied)
    target.sync()
    return target
