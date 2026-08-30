
from __future__ import annotations

import xml.etree.ElementTree as ET


def _text(node,tag):
    child=node.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _attr(node,name):
    value=node.attrib.get(name)
    return str(value).strip() if value is not None else ""


def iter_verbnet(path=None):
    """
    Read VerbNet through NLTK when path is None.

    VerbNet exposes structured verb classes with:
      members
      thematic roles + selectional restrictions
      syntactic frames
      semantic predicates

    The structures are kept typed rather than flattened into generic edges.
    """
    from nltk.corpus import verbnet as vn

    # Trigger corpus loading and give the caller a useful missing-corpus error.
    try:
        classes=vn.classids()
    except LookupError as exc:
        raise RuntimeError(
            "NLTK VerbNet corpus is missing. Install/download it first with "
            "`python -c \"import nltk; nltk.download('verbnet')\"`."
        ) from exc

    for class_id in classes:
        try:
            root=vn.vnclass(class_id)
        except Exception:
            continue

        if root is None:
            continue

        members=root.find("MEMBERS")
        if members is not None:
            for member in members.findall("MEMBER"):
                name=_attr(member,"name")
                if name:
                    yield {
                        "kind":"member",
                        "class_id":class_id,
                        "verb":name,
                        "metadata":{
                            "wn":_attr(member,"wn"),
                            "fn":_attr(member,"fn"),
                            "framenet":_attr(member,"framenet"),
                        },
                    }

        roles=root.find("THEMROLES")
        if roles is not None:
            for role in roles.findall("THEMROLE"):
                role_type=_attr(role,"type")
                restrictions=[]

                sel=role.find("SELRESTRS")
                if sel is not None:
                    for r in sel.findall("SELRESTR"):
                        restrictions.append({
                            "type":_attr(r,"type"),
                            "value":_attr(r,"Value"),
                        })

                yield {
                    "kind":"role",
                    "class_id":class_id,
                    "role":role_type,
                    "restrictions":restrictions,
                }

        frames=root.find("FRAMES")
        if frames is not None:
            for frame_index,frame in enumerate(frames.findall("FRAME")):
                example=_text(frame,"EXAMPLE")

                syntax=[]
                syntax_node=frame.find("SYNTAX")
                if syntax_node is not None:
                    for child in list(syntax_node):
                        syntax.append({
                            "tag":child.tag,
                            "value":_attr(child,"value"),
                            "type":_attr(child,"type"),
                            "selrestrs":[
                                {
                                    "type":_attr(x,"type"),
                                    "value":_attr(x,"Value"),
                                }
                                for x in child.findall("SELRESTRS/SELRESTR")
                            ],
                        })

                semantics=[]
                sem_node=frame.find("SEMANTICS")
                if sem_node is not None:
                    for pred in sem_node.findall("PRED"):
                        arguments=[
                            {
                                "value":_attr(arg,"value"),
                                "type":_attr(arg,"type"),
                            }
                            for arg in pred.findall("ARGS/ARG")
                        ]
                        semantics.append({
                            "value":_attr(pred,"value"),
                            "bool":_attr(pred,"bool"),
                            "arguments":arguments,
                        })

                yield {
                    "kind":"frame",
                    "class_id":class_id,
                    "frame_index":frame_index,
                    "description":_attr(frame,"description"),
                    "example":example,
                    "syntax":syntax,
                    "semantics":semantics,
                }


def ensure_verbnet(nltk_download=False):
    try:
        from nltk.corpus import verbnet
        verbnet.classids()
        return
    except LookupError:
        if not nltk_download:
            raise

    import nltk
    nltk.download("verbnet")
    from nltk.corpus import verbnet
    verbnet.classids()
