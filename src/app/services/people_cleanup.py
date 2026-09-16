from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Iterable

from infra.cancel import raise_if_cancelled

from .face_search import FaceClusterMember, FaceIndexService, FaceRegionLabelMutationResult
from .library_catalog import CatalogQuery, LibraryCatalogService


@dataclass(frozen=True)
class PeopleCleanupGroup:
    group_id: str
    members: tuple[FaceClusterMember, ...]
    suggested_name: str = ""
    suggested_score: float = 0.0
    kind: str = "similar"
    priority: float = 0.0
    summary: str = ""


@dataclass(frozen=True)
class PeopleCleanupSnapshot:
    groups: tuple[PeopleCleanupGroup, ...]
    total_visible_faces: int
    labeled_faces: int
    unlabelled_faces: int
    hidden_faces: int

    @property
    def completion_percent(self) -> int:
        eligible = max(0, self.total_visible_faces)
        return int(round(100 * self.labeled_faces / eligible)) if eligible else 100


class PeopleCleanupService:
    """Global registered-root curation feed built from durable face records."""

    def __init__(self, face_service: FaceIndexService, *, catalog: LibraryCatalogService | None = None) -> None:
        self.face_service = face_service
        self.catalog = catalog or LibraryCatalogService()

    def build_snapshot(
        self,
        *,
        root_ids: Iterable[str] = (),
        scope_paths: Iterable[str] | None = None,
        backend: str = "cosine-kmeans",
        min_face_score: float = 0.0,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check=None,
    ) -> PeopleCleanupSnapshot:
        paths = self._catalog_paths(root_ids, scope_paths=scope_paths)
        if progress_callback:
            progress_callback(0, "Loading registered-library faces")
        records = self.face_service.load_all_records(candidate_paths=paths, include_tiny_faces=False)
        raise_if_cancelled(cancel_check)
        visible = [record for record in records if not bool(record.hidden)]
        labeled = [record for record in visible if str(record.person_name or "").strip()]
        unlabelled = [record for record in visible if not str(record.person_name or "").strip()]
        if len(unlabelled) < 2:
            return PeopleCleanupSnapshot((), len(visible), len(labeled), len(unlabelled), len(records) - len(visible))
        if progress_callback:
            progress_callback(18, "Grouping unlabelled faces for review")
        refs = [(record.image_path, int(record.face_index)) for record in unlabelled]
        feedback_refs = [self._member_ref(record) for record in unlabelled]
        requested_clusters = min(max(2, int(round(sqrt(len(refs))))), len(refs))
        comparison = self.face_service.cluster_faces_compare(
            num_clusters=requested_clusters,
            min_face_score=float(min_face_score),
            backends=[str(backend)],
            outlier_policy="isolate",
            candidate_paths=paths,
            face_refs=refs,
            include_tiny_faces=False,
            cancel_check=cancel_check,
        )
        clusters = dict(comparison.clusters_by_key.get(str(backend), {}) or {})
        suggestions = dict(comparison.identity_suggestions_by_key.get(str(backend), {}) or {})
        feedback = self.catalog.face_cleanup_feedback(feedback_refs)
        exclusions = self.catalog.face_cleanup_exclusions(feedback_refs)
        groups: list[PeopleCleanupGroup] = []
        for cluster_id, members in clusters.items():
            raise_if_cancelled(cancel_check)
            members = tuple(member for member in members if not bool(member.hidden) and not str(member.person_name or "").strip())
            if not members:
                continue
            separated = self._separated_members(members, exclusions)
            accepted_members = tuple(member for member in members if member not in separated)
            suggested = suggestions.get(int(cluster_id))
            suggestion_name = str(getattr(suggested, "person_name", "") or "").strip()
            if suggestion_name and all(feedback.get((*self._member_ref(member), suggestion_name)) == "rejected" for member in accepted_members):
                suggestion_name = ""
            if accepted_members:
                kind = "singleton" if int(cluster_id) == -1 or len(accepted_members) == 1 else "similar"
                score = float(getattr(suggested, "mean_score", 0.0) or 0.0) if suggestion_name else 0.0
                priority = float(len(accepted_members)) * 100.0 + score * 10.0 + (50.0 if suggestion_name else 0.0)
                groups.append(
                    PeopleCleanupGroup(
                        group_id=f"{backend}:{int(cluster_id)}",
                        members=accepted_members,
                        suggested_name=suggestion_name,
                        suggested_score=score,
                        kind=kind,
                        priority=priority,
                        summary=self._summary(kind, len(accepted_members), suggestion_name, score),
                    )
                )
            for member in separated:
                groups.append(
                    PeopleCleanupGroup(
                        group_id=f"{backend}:{int(cluster_id)}:split:{member.image_path}:{int(member.face_index)}",
                        members=(member,), kind="split", priority=1.0,
                        summary="Kept separate by an earlier Split decision.",
                    )
                )
        groups.sort(key=lambda item: (-item.priority, item.group_id))
        if progress_callback:
            progress_callback(100, f"Prepared {len(groups)} people cleanup groups")
        return PeopleCleanupSnapshot(tuple(groups), len(visible), len(labeled), len(unlabelled), len(records) - len(visible))

    def apply_name(
        self,
        person_name: str,
        members: Iterable[FaceClusterMember],
        *,
        progress=None,
        cancel_check=None,
    ) -> FaceRegionLabelMutationResult:
        refs = [(member.image_path, int(member.face_index)) for member in members]
        return self.face_service.label_unlabeled_indexed_faces_with_metadata(
            str(person_name), refs, source="people_cleanup_inbox", progress=progress, cancel_check=cancel_check,
        )

    def reject_suggestion(
        self,
        person_name: str,
        members: Iterable[FaceClusterMember],
        *,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check=None,
    ) -> None:
        members = tuple(members)
        for index, member in enumerate(members, start=1):
            raise_if_cancelled(cancel_check)
            self.catalog.set_face_cleanup_feedback(member.image_path, int(member.face_index), str(person_name), "rejected")
            if progress_callback:
                progress_callback(int(index * 100 / max(1, len(members))), f"Rejecting suggestion for face {index}/{len(members)}")

    def split_members(self, selected: Iterable[FaceClusterMember], group_members: Iterable[FaceClusterMember]) -> None:
        selected_refs = {(member.image_path, int(member.face_index)) for member in selected}
        for left in selected_refs:
            for member in group_members:
                right = (member.image_path, int(member.face_index))
                if right not in selected_refs:
                    self.catalog.add_face_cleanup_exclusion(left, right)

    def split_group(
        self,
        members: Iterable[FaceClusterMember],
        *,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check=None,
    ) -> None:
        """Persist every pairwise exclusion so a review group becomes singletons."""
        refs = sorted({(member.image_path, int(member.face_index)) for member in members})
        total = max(1, len(refs) * (len(refs) - 1) // 2)
        completed = 0
        for index, left in enumerate(refs):
            for right in refs[index + 1 :]:
                raise_if_cancelled(cancel_check)
                self.catalog.add_face_cleanup_exclusion(left, right)
                completed += 1
                if progress_callback and (completed == total or completed % 24 == 0):
                    progress_callback(int(completed * 100 / total), f"Saving split decisions {completed}/{total}")

    def hide_members(
        self,
        members: Iterable[FaceClusterMember],
        *,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check=None,
    ) -> None:
        members = tuple(members)
        for index, member in enumerate(members, start=1):
            raise_if_cancelled(cancel_check)
            self.face_service.hide_face(member.image_path, int(member.face_index))
            if progress_callback:
                progress_callback(int(index * 100 / max(1, len(members))), f"Hiding face {index}/{len(members)}")

    def merge_people(self, source_name: str, target_name: str) -> None:
        self.face_service.merge_person_identities(source_name, target_name)

    def _catalog_paths(self, root_ids: Iterable[str], *, scope_paths: Iterable[str] | None = None) -> list[str]:
        offset = 0
        result: list[str] = []
        roots = tuple(str(root) for root in root_ids if str(root))
        roots_scope = tuple(str(item) for item in scope_paths if str(item)) if scope_paths is not None else None
        while True:
            page = self.catalog.query_assets(CatalogQuery(root_ids=roots, scope_paths=roots_scope, order="captured_asc", offset=offset, limit=1000))
            result.extend(asset.image_path for asset in page.items)
            if page.next_offset is None:
                return result
            offset = page.next_offset

    def _separated_members(self, members, exclusions):
        refs = {self._member_ref(member) for member in members}
        separated_refs = {left for left, right in exclusions if left in refs and right in refs}
        separated_refs.update(right for left, right in exclusions if left in refs and right in refs)
        return tuple(member for member in members if self._member_ref(member) in separated_refs)

    def _member_ref(self, member: FaceClusterMember) -> tuple[str, int]:
        return self.catalog.canonical_path(member.image_path), int(member.face_index)

    @staticmethod
    def _summary(kind: str, count: int, suggestion_name: str, score: float) -> str:
        if suggestion_name:
            return f"{count} similar unlabelled face{'s' if count != 1 else ''}; possible {suggestion_name} ({score:.3f})."
        if kind == "singleton":
            return "One unlabelled face needs manual review."
        return f"{count} similar unlabelled faces ready for review."
