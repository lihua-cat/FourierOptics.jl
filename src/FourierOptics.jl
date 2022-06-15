module FourierOptics

using FFTW
using Images
using CUDA
import Colors: colormatch, xyY

include("defaults.jl")
include("utils.jl")

export aperture
include("apertures.jl")

export CoherentField2D, CoherentField1D
export intensity, power, get_color
include("light_sources.jl")

export propagatorTF, propagatorIR, propTF!, propIR!
include("propagation.jl")

export illuminantD65, illuminantD
include("illuminant.jl")

export lens
include("lens.jl")

end