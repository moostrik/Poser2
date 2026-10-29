"""Which OAK backend the installed depthai supports: `oak` (depthai 2, the OAK-D) or `oak4`
(depthai 3, the OAK4). The two are one package name, so a PC holds one of them and the install
is the choice (`scripts\\install.bat`, `scripts\\install.bat oak4`); an app takes its camera
classes from the backend named here and never imports both."""

from importlib.metadata import version, PackageNotFoundError

OAK_BACKENDS: dict[int, str] = {2: 'oak', 3: 'oak4'}


def oak_backend(depthai_version: str | None = None) -> str:
    """The backend package name for the installed depthai, read from the package metadata so
    nothing is imported: `oak` for 2.x, `oak4` for 3.x. Any other install is an error that names
    the fix."""
    if depthai_version is None:
        try:
            depthai_version = version('depthai')
        except PackageNotFoundError:
            raise ImportError('depthai is not installed: run scripts\\install.bat (or install.bat oak4 for an OAK4 PC)') from None
    try:
        major: int = int(depthai_version.split('.')[0])
    except ValueError:
        major = 0
    backend: str | None = OAK_BACKENDS.get(major)
    if backend is None:
        raise ImportError(f'depthai {depthai_version} fits no backend: 2.x is oak (scripts\\install.bat), 3.x is oak4 (install.bat oak4)')
    return backend


def require_backend(name: str) -> None:
    """Raise at import when the installed depthai is not the one backend `name` runs on, with the
    line that fixes the install."""
    installed: str = oak_backend()
    if installed != name:
        raise ImportError(f'the {name} backend needs depthai {[k for k, v in OAK_BACKENDS.items() if v == name][0]}, '
                          f'but depthai {version("depthai")} is installed (the {installed} backend): '
                          f'run scripts\\install.bat{" oak4" if name == "oak4" else ""}')
