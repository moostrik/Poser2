"""Which OAK backend the installed depthai supports: `oak4` (depthai 3, the OAK4, `.venv`) or `oak` (depthai 2,
the OAK-D, `.venv-oakd`); one venv holds one of them."""

from importlib.metadata import version, PackageNotFoundError

OAK_BACKENDS: dict[int, str] = {2: 'oak', 3: 'oak4'}
VENVS: dict[str, str] = {'oak': '.venv-oakd', 'oak4': '.venv'}


def oak_backend(depthai_version: str | None = None) -> str:
    """The backend package name for the installed depthai, read from the package metadata so
    nothing is imported: `oak` for 2.x, `oak4` for 3.x. Any other install is an error that names
    the fix."""
    if depthai_version is None:
        try:
            depthai_version = version('depthai')
        except PackageNotFoundError:
            raise ImportError('depthai is not installed: run scripts\\install.bat (.venv) or scripts\\install.bat oakd (.venv-oakd)') from None
    try:
        major: int = int(depthai_version.split('.')[0])
    except ValueError:
        major = 0
    backend: str | None = OAK_BACKENDS.get(major)
    if backend is None:
        raise ImportError(f'depthai {depthai_version} fits no backend: 3.x is oak4 (.venv, scripts\\install.bat), 2.x is oak (.venv-oakd, install.bat oakd)')
    return backend


def require_backend(name: str) -> None:
    """Raise at import when the installed depthai is not the one backend `name` runs on, with the
    venv that has it."""
    installed: str = oak_backend()
    if installed != name:
        raise ImportError(f'the {name} backend needs depthai {[k for k, v in OAK_BACKENDS.items() if v == name][0]}, '
                          f'but depthai {version("depthai")} is installed (the {installed} backend): '
                          f'start it with {VENVS[name]}\\Scripts\\python (scripts\\install.bat{" oakd" if name == "oak" else ""} builds it)')
