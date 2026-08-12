"""Static, path-free capability catalog for production engine adapters.

Cataloging is deliberately independent of model packages.  Importing this
module, listing capabilities, and checking whether a model is enabled must be
safe in the application process where no isolated engine environment exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any


CHATTERBOX_ENGINE_ID = "chatterbox"
CHATTERBOX_NANO_MODEL_ID = "nano"
CHATTERBOX_SOURCE_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
CHATTERBOX_PACKAGE = "chatterbox-tts"
CHATTERBOX_PACKAGE_VERSION = "0.1.7"
CHATTERBOX_NANO_MODEL = "ResembleAI/chatterbox-nano"


@dataclass(frozen=True)
class EngineModelCapability:
    """Immutable, path-free description of one engine/model combination."""

    engine_id: str
    model_id: str
    display_name: str
    language: str
    languages: tuple[str, ...]
    package: str
    package_version: str
    model: str
    model_revision: str
    model_checksum: str
    reference_wav_required: bool
    fixed_speed: float
    chunk_cap: int
    runtime: str
    enabled: bool
    watermark_notice: str

    @property
    def id(self) -> str:
        return f"{self.engine_id}.{self.model_id}"

    @property
    def requires_reference_wav(self) -> bool:
        """Compatibility spelling for callers describing input requirements."""

        return self.reference_wav_required

    @property
    def speed(self) -> float:
        """Fixed initial speed exposed with the familiar settings spelling."""

        return self.fixed_speed

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["id"] = self.id
        return result


_NANO_NOTICE = "Chatterbox output includes the applicable model watermark notice."
_DISABLED_NOTICE = "Model is cataloged for future evaluation and is not enabled."

_CAPABILITIES = (
    EngineModelCapability(
        engine_id=CHATTERBOX_ENGINE_ID,
        model_id=CHATTERBOX_NANO_MODEL_ID,
        display_name="Chatterbox Nano",
        language="en",
        languages=("en",),
        package=CHATTERBOX_PACKAGE,
        package_version=CHATTERBOX_PACKAGE_VERSION,
        model=CHATTERBOX_NANO_MODEL,
        model_revision=CHATTERBOX_SOURCE_COMMIT,
        model_checksum="unrecorded",
        reference_wav_required=False,
        fixed_speed=1.0,
        chunk_cap=300,
        runtime="cpu",
        enabled=True,
        watermark_notice=_NANO_NOTICE,
    ),
    EngineModelCapability(
        engine_id=CHATTERBOX_ENGINE_ID,
        model_id="turbo",
        display_name="Chatterbox Turbo",
        language="en",
        languages=("en",),
        package=CHATTERBOX_PACKAGE,
        package_version=CHATTERBOX_PACKAGE_VERSION,
        model="ResembleAI/chatterbox-turbo",
        model_revision="unresolved",
        model_checksum="unrecorded",
        reference_wav_required=True,
        fixed_speed=1.0,
        chunk_cap=300,
        runtime="cpu",
        enabled=False,
        watermark_notice=_DISABLED_NOTICE,
    ),
    EngineModelCapability(
        engine_id=CHATTERBOX_ENGINE_ID,
        model_id="base",
        display_name="Chatterbox Base",
        language="en",
        languages=("en",),
        package=CHATTERBOX_PACKAGE,
        package_version=CHATTERBOX_PACKAGE_VERSION,
        model="ResembleAI/chatterbox",
        model_revision="unresolved",
        model_checksum="unrecorded",
        reference_wav_required=True,
        fixed_speed=1.0,
        chunk_cap=300,
        runtime="cpu",
        enabled=False,
        watermark_notice=_DISABLED_NOTICE,
    ),
    EngineModelCapability(
        engine_id=CHATTERBOX_ENGINE_ID,
        model_id="multilingual",
        display_name="Chatterbox Multilingual",
        language="multilingual",
        languages=("multilingual",),
        package=CHATTERBOX_PACKAGE,
        package_version=CHATTERBOX_PACKAGE_VERSION,
        model="ResembleAI/chatterbox-multilingual",
        model_revision="unresolved",
        model_checksum="unrecorded",
        reference_wav_required=True,
        fixed_speed=1.0,
        chunk_cap=300,
        runtime="cpu",
        enabled=False,
        watermark_notice=_DISABLED_NOTICE,
    ),
)

_BY_KEY = MappingProxyType({(entry.engine_id, entry.model_id): entry for entry in _CAPABILITIES})


def _revision_payload() -> bytes:
    return json.dumps(
        [entry.as_dict() for entry in _CAPABILITIES],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


CATALOG_REVISION = hashlib.sha256(_revision_payload()).hexdigest()


def catalog_revision() -> str:
    """Return the stable digest of all catalog records."""

    return CATALOG_REVISION


def list_capabilities() -> tuple[EngineModelCapability, ...]:
    """Return the ordered immutable catalog without importing model packages."""

    return _CAPABILITIES


def list_models(engine_id: str = CHATTERBOX_ENGINE_ID) -> tuple[EngineModelCapability, ...]:
    """Return immutable model entries for one engine."""

    if not isinstance(engine_id, str) or not engine_id:
        raise ValueError("engine id is required")
    entries = tuple(entry for entry in _CAPABILITIES if entry.engine_id == engine_id)
    if not entries:
        raise ValueError(f"unknown engine: {engine_id}")
    return entries


def get_capability(engine_id: str, model_id: str) -> EngineModelCapability:
    """Look up a catalog entry, including disabled placeholders."""

    try:
        return _BY_KEY[(engine_id, model_id)]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown engine/model: {engine_id}/{model_id}") from exc


def require_enabled(engine_id: str, model_id: str) -> EngineModelCapability:
    """Look up and require an enabled engine/model entry."""

    entry = get_capability(engine_id, model_id)
    if not entry.enabled:
        raise ValueError(f"engine/model is disabled: {engine_id}/{model_id}")
    return entry


def get_enabled_capability(engine_id: str, model_id: str) -> EngineModelCapability:
    """Compatibility spelling for the enabled lookup boundary."""

    return require_enabled(engine_id, model_id)


__all__ = [
    "CATALOG_REVISION",
    "CHATTERBOX_ENGINE_ID",
    "CHATTERBOX_NANO_MODEL",
    "CHATTERBOX_NANO_MODEL_ID",
    "CHATTERBOX_PACKAGE",
    "CHATTERBOX_PACKAGE_VERSION",
    "CHATTERBOX_SOURCE_COMMIT",
    "EngineModelCapability",
    "catalog_revision",
    "get_capability",
    "get_enabled_capability",
    "list_capabilities",
    "list_models",
    "require_enabled",
]
