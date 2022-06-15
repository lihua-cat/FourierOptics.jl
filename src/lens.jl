function lens(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    λ::Real,
    f::Real;
    origin=middle.((x, y)),
    gpu=false
)
    T = Complex{eltype(x)}
    p = Matrix{T}(undef, length(x), length(y))
    x0, y0 = origin
    x = x .- x0
    y = y .- y0
    yt = transpose(y)
    r2 = @. x^2 + yt^2
    @. p = cispi(-r2 / λ / f)
    if gpu
        return CuArray(p)
    else
        return p
    end
end
function lens(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    λ::AbstractVector{<:Real},
    f::AbstractVector{<:Real};
    origin=middle.((x, y)),
    gpu=false
)
    @assert length(λ) == length(f)
    T = Complex{eltype(x)}
    p = Array{T}(undef, length(x), length(y), length(λ))
    for i in 1:length(λ)
        p[:, :, i] = lens(x, y, λ[i], f[i];origin=origin, gpu=false)
    end
    if gpu
        return CuArray(p)
    else
        return p
    end
end
function lens(u::CoherentField2D, f;origin=nothing, gpu=false)
    (;x, y, λ) = u
    if isnothing(origin)
        origin = middle.((x, y))
    end
    f = f .* ones(length(λ))
    lens(x, y, λ, f;origin=origin, gpu=gpu)
end