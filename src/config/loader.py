from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.errors import ConfigNotFoundError, ConfigValidationError
from config.models import RecipeConfig, ResolvedExperiment, StageDefinition
from config.registry import ConfigRegistry, build_default_registry


class ConfigLoader:
    def __init__(
        self,
        config_root: str | Path = "config",
        registry: ConfigRegistry | None = None,
    ) -> None:
        self.config_root = Path(config_root)
        self.registry = registry or build_default_registry()

    def resolve(
        self,
        recipe_name: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> ResolvedExperiment:
        recipe_raw, recipe_path = self._load_recipe_raw(recipe_name)

        refs_overrides, non_ref_overrides = self._split_overrides(overrides or {})
        if refs_overrides:
            self._apply_overrides(recipe_raw, refs_overrides)

        raw_refs = self._mapping_from(recipe_raw, "refs")
        raw_run = self._mapping_from(recipe_raw, "run")
        raw_stages = self._mapping_from(recipe_raw, "stages")

        composite_raw: dict[str, Any] = {
            "name": recipe_raw.get("name", recipe_name),
            "description": recipe_raw.get("description"),
            "refs": dict(raw_refs),
            "run": dict(raw_run),
            "stages": dict(raw_stages),
        }

        refs = composite_raw["refs"]
        if not isinstance(refs, dict):
            raise ConfigValidationError("recipe.refs", "Expected a table/object")

        source_files: dict[str, Path] = {"recipe": recipe_path}
        for domain_name, profile_name in refs.items():
            if not isinstance(profile_name, str) or not profile_name.strip():
                raise ConfigValidationError(
                    "recipe.refs",
                    f"Ref '{domain_name}' must point to a non-empty profile name",
                )
            domain_raw, domain_path = self._load_domain_raw(domain_name, profile_name)
            composite_raw[domain_name] = domain_raw
            source_files[domain_name] = domain_path

        if non_ref_overrides:
            self._apply_overrides(composite_raw, non_ref_overrides)

        recipe = self._parse_recipe(composite_raw)
        model = self.registry.parse_domain("model", recipe.refs["model"], self._mapping_from(composite_raw, "model"))
        tracking = self.registry.parse_domain(
            "tracking",
            recipe.refs["tracking"],
            self._mapping_from(composite_raw, "tracking"),
        )

        data = None
        if "data" in recipe.refs:
            data = self.registry.parse_domain("data", recipe.refs["data"], self._mapping_from(composite_raw, "data"))

        metric = None
        if "metric" in recipe.refs:
            metric = self.registry.parse_domain(
                "metric",
                recipe.refs["metric"],
                self._mapping_from(composite_raw, "metric"),
            )

        reward = None
        if "reward" in recipe.refs:
            reward = self.registry.parse_domain(
                "reward",
                recipe.refs["reward"],
                self._mapping_from(composite_raw, "reward"),
            )

        return ResolvedExperiment(
            recipe=recipe,
            model=model,
            tracking=tracking,
            data=data,
            metric=metric,
            reward=reward,
            source_files=source_files,
        )

    def _load_recipe_raw(self, recipe_name: str) -> tuple[dict[str, Any], Path]:
        path = self.config_root / "recipe" / f"{recipe_name}.toml"
        return self._load_toml(path), path

    def _load_domain_raw(self, domain_name: str, profile_name: str) -> tuple[dict[str, Any], Path]:
        domain_spec = self.registry.get_domain(domain_name)
        path = self.config_root / domain_spec.directory / f"{profile_name}.toml"
        return self._load_toml(path), path

    def _load_toml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ConfigNotFoundError(path)
        with path.open("rb") as handle:
            return tomllib.load(handle)

    def _parse_recipe(self, raw: Mapping[str, Any]) -> RecipeConfig:
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigValidationError("recipe", "'name' must be a non-empty string")

        description = raw.get("description")
        if description is not None and not isinstance(description, str):
            raise ConfigValidationError("recipe", "'description' must be a string")

        refs = self._mapping_from(raw, "refs")
        if "model" not in refs:
            raise ConfigValidationError("recipe.refs", "Recipe must declare a 'model' ref")
        if "tracking" not in refs:
            raise ConfigValidationError("recipe.refs", "Recipe must declare a 'tracking' ref")

        run = self.registry.parse_run(self._mapping_from(raw, "run"))

        stages_raw = self._mapping_from(raw, "stages")
        if not stages_raw:
            raise ConfigValidationError("recipe.stages", "Recipe must define at least one stage")

        stages: list[StageDefinition] = []
        for stage_name in stages_raw:
            if not isinstance(stage_name, str) or not stage_name.strip():
                raise ConfigValidationError("recipe.stages", "Stage names must be non-empty strings")
            stage_mapping = self._mapping_from(stages_raw, stage_name)
            stage_config = self.registry.parse_stage(stage_name, stage_mapping)
            stages.append(StageDefinition(name=stage_name, config=stage_config))

        return RecipeConfig(
            name=name,
            description=description,
            refs=self._validate_refs(refs),
            run=run,
            stages=tuple(stages),
        )

    def _mapping_from(self, raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = raw.get(key, {})
        if not isinstance(value, Mapping):
            raise ConfigValidationError(key, "Expected a table/object")
        return value

    def _validate_refs(self, refs: Mapping[str, Any]) -> dict[str, str]:
        validated: dict[str, str] = {}
        for key, value in refs.items():
            if not isinstance(key, str) or not key.strip():
                raise ConfigValidationError("recipe.refs", "Ref keys must be non-empty strings")
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError("recipe.refs", f"Ref '{key}' must be a non-empty string")
            validated[key] = value
        return validated

    def _split_overrides(
        self,
        overrides: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        refs_overrides: dict[str, Any] = {}
        non_ref_overrides: dict[str, Any] = {}
        for path, value in overrides.items():
            if path == "refs" or path.startswith("refs."):
                refs_overrides[path] = value
            else:
                non_ref_overrides[path] = value
        return refs_overrides, non_ref_overrides

    def _apply_overrides(self, target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
        for path, value in overrides.items():
            if not isinstance(path, str) or not path.strip():
                raise ConfigValidationError("overrides", "Override keys must be non-empty dot paths")

            keys = path.split(".")
            current: dict[str, Any] = target
            for key in keys[:-1]:
                nested = current.get(key)
                if nested is None:
                    nested = {}
                    current[key] = nested
                if not isinstance(nested, dict):
                    raise ConfigValidationError("overrides", f"Cannot descend into non-table path '{path}'")
                current = nested
            current[keys[-1]] = value
