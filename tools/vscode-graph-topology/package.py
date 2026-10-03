"""Pack the extension into a .vsix, with nothing but the standard library
(a .vsix is a zip: a manifest, a content-types file, the extension).

    python tools/vscode-graph-topology/package.py            -> dist/*.vsix
    python tools/vscode-graph-topology/package.py --install  and install it
                                                             (`code` CLI)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
FILES = ("package.json", "extension.js", "lib.js", "README.md")

CONTENT_TYPES = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension=".json" ContentType="application/json"/>
  <Default Extension=".js" ContentType="application/javascript"/>
  <Default Extension=".md" ContentType="text/markdown"/>
  <Default Extension=".vsixmanifest" ContentType="text/xml"/>
</Types>
"""


def manifest(meta: dict) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Language="en-US" Id="{meta['name']}" Version="{meta['version']}" Publisher="{meta['publisher']}"/>
    <DisplayName>{escape(meta['displayName'])}</DisplayName>
    <Description xml:space="preserve">{escape(meta['description'])}</Description>
    <Categories>Other</Categories>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{meta['engines']['vscode']}"/>
    </Properties>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true"/>
  </Assets>
</PackageManifest>
"""


def build() -> Path:
    meta = json.loads((HERE / "package.json").read_text(encoding="utf-8"))
    DIST.mkdir(exist_ok=True)
    out = DIST / f"{meta['name']}-{meta['version']}.vsix"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as pack:
        pack.writestr("[Content_Types].xml", CONTENT_TYPES)
        pack.writestr("extension.vsixmanifest", manifest(meta))
        for name in FILES:
            pack.write(HERE / name, f"extension/{name}")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = build()
    print(f"-> {out}")
    if "--install" in argv:
        code = shutil.which("code")
        if code is None:
            print("no `code` on the PATH: install it from VS Code with "
                  "'Extensions: Install from VSIX...'")
            return 1
        subprocess.run([code, "--install-extension", str(out), "--force"],
                       check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
