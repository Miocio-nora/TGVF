from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbLogger:
    def __init__(
        self,
        *,
        project: str | None,
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str | None = None,
        tags: list[str] | None = None,
        directory: str | Path | None = None,
    ) -> None:
        self.enabled = bool(project)
        self.run = None
        self._wandb = None
        if not self.enabled:
            return
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "wandb is not installed. Install the tracking extra or `pip install wandb`."
            ) from exc
        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type=job_type,
            config=config or {},
            mode=mode,
            tags=tags,
            dir=str(directory) if directory else None,
        )

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(_jsonable(metrics), step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        for key, value in flatten_metrics(values).items():
            self.run.summary[key] = value

    def log_artifact(
        self,
        *,
        name: str,
        artifact_type: str,
        paths: list[str | Path],
        aliases: list[str] | None = None,
    ) -> None:
        if not self.enabled or self._wandb is None:
            return
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        added = False
        for path_like in paths:
            path = Path(path_like)
            if not path.exists():
                continue
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            added = True
        if added:
            self._wandb.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


def flatten_metrics(
    values: dict[str, Any],
    *,
    prefix: str = "",
    sep: str = "/",
) -> dict[str, int | float | str | bool]:
    flattened: dict[str, int | float | str | bool] = {}
    for key, value in values.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=name, sep=sep))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            flattened[name] = "null" if value is None else value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (int, float, str, bool)) for item in value
        ):
            flattened[name] = ",".join(str(item) for item in value)
    return flattened


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
