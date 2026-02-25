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
from .InjectChannel3D import InjectChannel3D
from .Clamp3D import Clamp3D
from .Dampen3D import Dampen3D
from .Composite3D import Composite3D
from .Add3D import Add3D
from .Blit3D import Blit3D
from .InjectBinary3D import InjectBinary3D
