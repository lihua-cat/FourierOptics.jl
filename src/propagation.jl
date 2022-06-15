@doc raw"""
    propTF!(u, H, p)

Compute the diffraction field with the transfer function $H(\nu_x, \nu_y)$. 
`H` starts at $\nu=0$

```math
u_1(x, y, z=d) = \mathcal{F}^{-1}\left\{\mathcal{F}\{u_0(x, y, z=0)\}H(\nu_x, \nu_y)\right\}
```
"""
function propTF!(
    u::AbstractArray{Complex{T}}, 
    H::AbstractArray{Complex{T}}, 
    p::AbstractFFTs.Plan
) where {T<:AbstractFloat}
    # u .= ifftshift(u)
    p * u           # fft!(u)
    u .*= H         # u * H
    p \ u           # ifft!(u)
    # u .= fftshift(u)
    u
end

@doc raw"""
    propIR!(u, h, ds, p)

Compute the diffraction field with the impulse response $h(x, y)$. 
`u` starts at $x=0$, `h` starts at $x=0$

```math
u_1(x, y, z=d) = \mathcal{F}^{-1}\left\{\mathcal{F}\{u_0(x, y, z=0)\}\mathcal{F}\{h(x,y)\}\right\}
```
"""
function propIR!(
    u::AbstractArray{Complex{T}}, 
    h::AbstractArray{Complex{T}}, 
    ds::Number,
    p::AbstractFFTs.Plan
) where {T<:AbstractFloat}
    p * u
    p * h
    h *= ds
    u .*= h
    p \ u
    u
end

@doc raw"""
    propagatorTF(
        L::Union{Number, NTuple{2, Number}},
        N::Union{Integer, NTuple{2, Integer}},
        λ::Union{Number,AbstractVector{<:Number}},
        d::Union{Number,AbstractVector{<:Number}};
        method::Symbol=:ASM,
        gpu::Bool=false,
        T=$PRECISION
    )

Transfer function propagators used in convolution-based method.
The `method` includes `:ASM"` and `:Fresnel"`.

```math
H(\nu_x, \nu_y) = \left\{\begin{array}{ll} 
\exp\left[i\frac{2\pi d}{\lambda}\sqrt{1-(\lambda\nu_x)^2-(\lambda\nu_y)^2}\right] 
& \texttt{Angualr Spectrum or Rayleigh–Sommerfeld 1st solution} \\
e^{ikd} \exp\left[-i\pi \lambda d(\nu_x^2 + \nu_y^2)\right]
& \texttt{Fresnel approximation}
\end{array} \right.
```

# Arguments
- `L`: window width
- `N`: sample number
- `λ`: wavelength
- `d`: propagation distance
- `method`: `:ASM` or `:Fresnel`
- `gpu`: use gpu
- `T`: default $PRECISION

# Reference:
VOELZ, D. G. (2011). Computational Fourier optics. Bellingham, Wash, SPIE.
"""
function propagatorTF(
    L::Union{Number,NTuple{2,Number}},
    N::Union{Integer,NTuple{2,Integer}},
    λ::Union{Number,AbstractVector{<:Number}},
    d::Union{Number,AbstractVector{<:Number}};
    method::Symbol=:ASM,
    gpu::Bool=false,
    T=PRECISION
)
    fs = N ./ L
    dims_field = length(fs)
    ν = fftfreq.(N, fs)
    ν2 = chirp(ν)
    if length(λ) > 1 && length(d) > 1 && length(λ) != length(d)
        error("😿")
    end
    n_λd = max(length(λ), length(d))
    if n_λd > 0
        λ = reshape(λ .* ones(n_λd), ones(Int, dims_field)..., n_λd)
        d = reshape(d .* ones(n_λd), ones(Int, dims_field)..., n_λd)
    end
    if method == :ASM
        H = @. cispi(2 / λ * d * √(complex(1 - λ^2 * ν2)))
    elseif method == :Fresnel
        H = @. cispi(2 / λ * d - λ * d * ν2)
    else
        error("😿")
    end
    if gpu
        return CuArray{Complex{T}}(H)
    else
        return Array{Complex{T}}(H)
    end
end

@doc raw"""
    propagatorIR(
        L::Union{Number, NTuple{2, Number}}, 
        N::Union{Integer, NTuple{2, Integer}}, 
        λ::Union{Number,AbstractVector{<:Number}},
        d::Union{Number,AbstractVector{<:Number}};
        method::Symbol=:ASM,
        gpu::Bool=false,
        T=$PRECISION
    )

Impulse response propagators used in convolution-based methods. 
The `method` includes `:ASM` and `:Fresnel`.

```math
h(x, y) = \left\{\begin{array}{ll} 
\frac{d}{i\lambda}\frac{\exp(ikr)}{r^2}
& \texttt{Angualr Spectrum or Rayleigh–Sommerfeld 1st solution} \\
\frac{e^{ikd}}{i\lambda d} \exp\left[\frac{ik}{2d}(x^2+y^2)\right]
& \texttt{Fresnel approximation}
\end{array} \right.
```

# Arguments
- `L`: window width
- `N`: sample number
- `λ`: wavelength
- `d`: propagation distance
- `method`: `:ASM` or `:Fresnel`
- `gpu`: use gpu
- `T`: default $PRECISION

# Reference:
VOELZ, D. G. (2011). Computational Fourier optics. Bellingham, Wash, SPIE.
"""
function propagatorIR(
    L::Union{Number,NTuple{2,Number}},
    N::Union{Integer,NTuple{2,Integer}},
    λ::Number,
    d::Number;
    method::Symbol=:ASM,
    gpu::Bool=false,
    T=PRECISION
)
    dx = L ./ N
    x = @. range(-N ÷ 2, (N - 1) ÷ 2) * dx
    x2 = chirp(x)
    if length(λ) > 1 && length(d) > 1 && length(λ) != length(d)
        error("😿")
    end
    n_λd = max(length(λ), length(d))
    d = T.(d)
    if n_λd > 0
        λ = reshape(λ .* ones(T, n_λd), ones(Int, dims_field)..., n_λd)
        d = reshape(d .* ones(T, n_λd), ones(Int, dims_field)..., n_λd)
    end
    if method == :ASM
        r2 = @. d^2 + x2
        h = @. d / (im * λ) * cispi(2 / λ * √(r2)) / r2
    elseif method == :Fresnel
        h = @. cispi(2 / λ * d) / (im * λ * d) * cispi(1 / λ / d * x2)
    else
        error("😿")
    end
    if gpu
        return CuArray(H)
    else
        return H
    end
end

propagatorTF(u::CoherentField, d::Number; kwargs...) = propagatorTF(u.L, u.N, u.λ, d; kwargs...)
propagatorIR(u::CoherentField, d::Number; kwargs...) = propagatorIR(u.L, u.N, u.λ, d; kwargs...)