from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SharedSerializable(Protocol):
    def _shared_getstate(self) -> dict[str, Any]:
        ...


class SharedPicklerMixin:
    """
    Mixin for picklers to allow objects to specify different serializations for
    shared-model pickling.  Also usable as a standalone pickler.
    """

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, SharedSerializable) and not isinstance(obj, type):
            state = obj._shared_getstate()  # pyright: ignore
            return (obj.__class__.__new__, (obj.__class__,), state)
        else:
            return NotImplemented
