"""Models paged on and off the card (v705): what the server has loaded is
kept, but only what fits on the card is kept *there* -- the rest waits in
memory and is brought back the moment it is called.

A goal of itself, thought through, loaded eight decoders of 690 MB (the
maths decoder, the plan sayer, four sketchers, the base decoder, the
own-bug editor) and three encoders of 480 MB, and let none go: 7.8 GB of
the card's 8 after one turn, and with the debugging editor beside them
the card spilled into memory -- a turn of three hours. Each loader keeps
its model, as it should (loading one takes seconds); what none of them can
know is what the others hold.

    sweep(budget)   after each turn: every model is watched, and the least
                    lately used moved to memory until those on the card fit
    a call          of one in memory (`forward`) moves it back first, the
                    least lately used of the others moved off to make room

Only a model of `transformers` is paged, whole (not one inside another):
brought back when it is called, when it is asked to `generate`, and when
its `device` is asked -- what callers put its inputs on (`generate` asked
it of a model in memory, put the inputs there, and the call brought the
weights back to the card: two devices).
"""
from __future__ import annotations

import gc
import threading
import time
import weakref

#: what the models on the card may take together, in megabytes: the card's
#: 8 GB less what the server holds besides (spaCy's transformer, the
#: reader, work space for a call)
BUDGET = 3500
#: -- or what the process was started with: several servers sharing the
#: card, each a part of it (the loop's measure, `loop.SERVERS`)
import os as _os
BUDGET = int(_os.environ.get("V705_PAGING_BUDGET") or BUDGET)
#: what torch may keep on the card beyond what is allocated, in megabytes,
#: after a model writes
SPARE = 1000
#: what of the card torch may take at all, as a part of it: past it, what
#: torch keeps of freed blocks is given back and the call tried again, not
#: spilled into memory -- the card's 8 GB held 6.1 allocated and 8.6
#: reserved, the card at 23 W of 77 doing "100%", waiting on the spill.
#: The rest is the display's, and what is not torch's (spaCy's transformer
#: runs on cupy)
FRACTION = float(_os.environ.get("V705_CARD_FRACTION") or 0.86)
#: whether a model came onto the card since the last walk of every object
#: ([bool], set by what `install` puts in)
ARRIVED = [True]
#: {id: (the model, held weakly -- one its loader lets go of is gone --
#: and when it was last called)}
WATCHED: dict = {}
_LOCK = threading.RLock()


def _size(model) -> int:
    return sum(one.numel() * one.element_size()
               for one in list(model.parameters()) + list(model.buffers()))


def _quantized(model) -> bool:
    return bool(getattr(model, "is_loaded_in_4bit", False) or
                getattr(model, "is_loaded_in_8bit", False) or
                getattr(model, "is_quantized", False))


def _on_card(model) -> bool:
    try:
        return next(model.parameters()).device.type == "cuda"
    except StopIteration:
        return False


def _models() -> list:
    """[(the model, when last called)] of those still held."""
    out = []
    for key, (ref, used) in list(WATCHED.items()):
        model = ref()
        if model is None:
            WATCHED.pop(key, None)
        else:
            out.append((model, used))
    return out


def _make_room(torch, needed: int, keep) -> None:
    """The least lately used models on the card moved to memory until
    `needed` bytes more fit in the budget -- `keep` left where it is."""
    with _LOCK:
        held = [(used, model) for model, used in _models()
                if model is not keep and _on_card(model)]
        total = sum(_size(model) for _, model in held) + (
            _size(keep) if keep is not None and _on_card(keep) else 0)
        held.sort(key=lambda one: one[0])
        for _, model in held:
            if total + needed <= BUDGET * 2 ** 20:
                break
            if _quantized(model):
                # 4 bits (the 3B code model): bitsandbytes cannot move it
                # off the card -- it counts, and stays
                continue
            model.to("cpu")
            model.paged = True
            total -= _size(model)
        torch.cuda.empty_cache()


def _called(model, args=None, kwargs=None):
    """Before a watched model is called: back on the card where it was
    moved off, and marked as used now."""
    import torch
    with _LOCK:
        entry = WATCHED.get(id(model))
        if entry is not None:
            WATCHED[id(model)] = (entry[0], time.time())
        if getattr(model, "paged", False) and not _on_card(model):
            _make_room(torch, _size(model), model)
            model.to("cuda")
            model.paged = False
    return None


def watch(torch) -> int:
    """Every model of `transformers` this process holds on the card,
    watched (once each): how many are."""
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return 0
    with _LOCK:
        import torch.nn
        modules = [one for one in gc.get_objects()
                   if isinstance(one, torch.nn.Module)]
        found = [one for one in modules if isinstance(one, PreTrainedModel)]
        # one inside any other module is left where it is: its parts may be
        # called apart from it (the reader's BERT, inside its own net, is
        # asked `get_input_embeddings()(ids)` -- paged off, no call brought
        # it back, and every reading failed)
        inner = {id(child) for one in modules for child in one.children()}
        inner |= {id(child) for one in found for child in one.modules()
                  if child is not one}
        for one in found:
            if id(one) in WATCHED or id(one) in inner or not _on_card(one):
                continue
            _paged(one)
            WATCHED[id(one)] = (weakref.ref(one), time.time())
        return len(_models())


def adopt(model):
    """One model just loaded, watched from now, and room made for it on the
    card -- without walking every object (`watch`): what a loader that
    keeps a large model calls. The model, as given."""
    import torch
    if not torch.cuda.is_available() or not _on_card(model):
        return model
    with _LOCK:
        if id(model) not in WATCHED:
            _paged(model)
            WATCHED[id(model)] = (weakref.ref(model), time.time())
        _make_room(torch, 0, model)
    return model


def install() -> None:
    """Room made on the card as a model of `transformers` comes onto it --
    loaded there, or moved there -- not only after the turn: a turn of the
    conversation test loaded the editor, the ways estimator, the judge and
    the answer checker one after another, and three servers that each kept
    to 1000 MB between turns held 7.6 GB of the card's 8 during them. What
    is made room for is not watched until a sweep (it may be inside another
    module); what is moved off is only what a sweep found whole."""
    try:
        import torch
        from transformers import PreTrainedModel
    except ImportError:
        return
    if getattr(PreTrainedModel, "_v705_paging", False) or \
            not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(FRACTION)
    loading, moving = PreTrainedModel.from_pretrained.__func__, \
        PreTrainedModel.to

    def arrived(model):
        try:
            if _on_card(model) and id(model) not in WATCHED:
                ARRIVED[0] = True
                # a model that writes is never a part of another's net (the
                # reader's BERT is one): watched now, so that the next to
                # come can make room by moving it -- a goal is one turn,
                # and seven decoders loaded in it none could move, the
                # card's 8 GB held 7.7 (16.5 asked of it)
                if model.can_generate():
                    adopt(model)
                else:
                    _make_room(torch, 0, model)
        except Exception:                           # noqa: BLE001
            pass
        return model

    def from_pretrained(cls, *args, **kwargs):
        try:
            return arrived(loading(cls, *args, **kwargs))
        except torch.cuda.OutOfMemoryError:
            # no room to load it: every model watched moved off, and
            # loaded once more
            _make_room(torch, 2 ** 50, None)
            return arrived(loading(cls, *args, **kwargs))

    def to(self, *args, **kwargs):
        return arrived(moving(self, *args, **kwargs))

    def cuda(self, *args, **kwargs):
        return arrived(onto(self, *args, **kwargs))
    onto = PreTrainedModel.cuda
    PreTrainedModel.from_pretrained = classmethod(from_pretrained)
    PreTrainedModel.to = to
    PreTrainedModel.cuda = cuda
    PreTrainedModel._v705_paging = True


def _paged(model) -> None:
    """A model brought back to the card first whenever it is called, asked
    to `generate`, or asked its `device`."""
    model.register_forward_pre_hook(_called)
    # (an encoder has no `generate`: BertModel, the reader's)
    generate = getattr(model, "generate", None)
    if generate is not None:
        def generating(*args, **kwargs):
            import torch
            _called(model)
            try:
                try:
                    return generate(*args, **kwargs)
                except torch.cuda.OutOfMemoryError:
                    # out of its part of the card: every other model moved
                    # off, and tried once more
                    _make_room(torch, 2 ** 50, model)
                    return generate(*args, **kwargs)
            finally:
                # what a batch's cache took, given back where it is more
                # than the card spares: kept by torch for the next, it is
                # room no other model can be moved into
                if torch.cuda.memory_reserved() > \
                        torch.cuda.memory_allocated() + SPARE * 2 ** 20:
                    torch.cuda.empty_cache()
        model.generate = generating
    kind = type(model)
    model.__class__ = type(kind.__name__, (kind,), {
        "device": property(lambda self: (_called(self),
                                         kind.device.fget(self))[1]),
        "__module__": kind.__module__})


def _transformers_model():
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return None
    return PreTrainedModel


def sweep(budget: int | None = None) -> dict:
    """After a turn: every model watched, and those on the card made to fit
    the budget. What is where, in megabytes."""
    import sys
    if "torch" not in sys.modules:
        return {}
    import torch
    if not torch.cuda.is_available():
        return {}
    global BUDGET
    if budget is not None:
        BUDGET = budget
    # every object walked only where a model came onto the card since the
    # last walk (`install` knows): a server of 8 GB walked whole took 22 s
    # after each turn, and after most no model has come
    if ARRIVED[0] or not getattr(_transformers_model(), "_v705_paging",
                                 False):
        ARRIVED[0] = False
        watch(torch)
    _make_room(torch, 0, None)
    with _LOCK:
        held = _models()
        on = sum(_size(model) for model, _ in held if _on_card(model))
        off = sum(_size(model) for model, _ in held if not _on_card(model))
    return {"on the card mb": round(on / 2 ** 20),
            "in memory mb": round(off / 2 ** 20), "watched": len(held)}
