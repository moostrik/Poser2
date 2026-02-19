"""3D fluid simulation compute shaders."""

from .Advect3D import Advect3D
from .Divergence3D import Divergence3D
from .Gradient3D import Gradient3D
from .JacobiPressure3D import JacobiPressure3D
from .JacobiDiffusion3D import JacobiDiffusion3D
from .VorticityCurl3D import VorticityCurl3D
from .VorticityForce3D import VorticityForce3D
from .Buoyancy3D import Buoyancy3D
from .Inject3D import Inject3D
from .Composite3D import Composite3D
from .Add3D import Add3D
