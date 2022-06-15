abstract type CoherentField{T,N} <: AbstractArray{T,N} end
Base.size(u::CoherentField{T,N}) where {T,N} = N==1 ? (u.N,) : u.N

struct CoherentField1D{T} <: CoherentField{T, 1}
    L::T
    N::Integer
    x::AbstractVector{T}
    λ::AbstractVector{T}
    E::AbstractMatrix{Complex{T}}
end

struct CoherentField2D{T} <: CoherentField{T, 2}
    L::NTuple{2,T}
    N::NTuple{2,Integer}
    x::AbstractVector{T}
    y::AbstractVector{T}
    λ::AbstractVector{T}
    E::AbstractArray{Complex{T},3}
end

Base.getindex(u::CoherentField1D, i) = (; x=u.x[i], E=u.E[i, :])
Base.getindex(u::CoherentField2D, i, j) = (; x=u.x[i], y=u.y[j], E=u.E[i, j, :])

function CoherentField1D(L, N, λ, E; gpu=false, T=PRECISION)
    d = L / N
    x = range(-N ÷ 2, (N - 1) ÷ 2) * d
    num_λ = length(λ)
    λ = λ .* ones(T, num_λ) # convert λ to [λ]
    if ndims(E) == 0
        E = E .* ones(N, num_λ)
    elseif ndims(E) == 1 && length(E) == num_λ
        E = reshape(E, (1, num_λ)) .* ones(N)
    elseif ndims(E) == 1 && length(E) == N
        E = E .* ones(1, num_λ)
    elseif ndims(E) == 2 && size(E) == (N, num_λ)
        nothing
    else
        error("😿")
    end
    E = Array{Complex{T}}(E)
    if gpu
        E = CuArray(E)
    end
    CoherentField1D{T}(L, N, x, λ, E)
end

function CoherentField2D(
    L::NTuple{2, Real}, 
    N::NTuple{2, Integer}, 
    λ::AbstractVector{<:Real}, 
    E::AbstractArray{<:Number}; 
    gpu=false, 
    T=PRECISION
)
    d = L ./ N
    x, y = @. range(-N ÷ 2, (N - 1) ÷ 2) * d
    shape = (N..., length(λ))
    E = _reshape_E(shape, E)
    if gpu
        E = CuArray{Complex{T}}(E)
    else
        E = Array{Complex{T}}(E)
    end
    CoherentField2D{T}(L, N, x, y, λ, E)
end

function _reshape_E(shape::NTuple{3,Integer}, E::AbstractArray{<:Number,3})
    @assert size(E) == shape
    E
end
function _reshape_E(shape::NTuple{3,Integer}, E::AbstractMatrix{<:Number})
    @assert size(E) == shape[1:2]
    E = repeat(E, 1, 1, shape[3])
end
function _reshape_E(shape::NTuple{3,Integer}, E::AbstractVector{<:Number})
    @assert length(E) == length(λ)
    E = repeat(reshape(E, (1, 1, shape[3])), shape[1:2]...)
end

intensity(u::CoherentField{T,N}) where {T,N} = dropdims(mapreduce(abs2, +, u.E, dims=N+1), dims=N+1)
power(u::CoherentField) = sum(intensity(u)) * prod(u.L ./ u.N)
function get_color(u::CoherentField{T,N}, max_intensity=1.0) where {T,N}
    I = Array(abs2.(u.E)) / max_intensity
    c = colormatch.(reshape(u.λ * 1e9, ones(Int, N)..., length(u.λ)))
    dropdims(sum(I .* c, dims=N+1), dims=N+1)
end

FFTW.plan_fft!(u::CoherentField{T, N}) where {T,N} = plan_fft!(u.E, Tuple(1:N))
function FFTW.fftfreq(u::CoherentField)
    (; L, N) = u
    d = L ./ N
    fs = 1 ./ d
    fftfreq.(N, fs)
end