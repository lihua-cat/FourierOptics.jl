const RIM_CONSTANT = 3
const RIM_WIDTH = 3

"""
    aperture(Nx::Integer, Ny::Integer, w; shape=:circ, origin=middle.((Nx, Ny)), rim=$RIM_WIDTH, T=$PRECISION)

Return a `Nx` \$\\times\$ `Ny` mask matrix of eltype `T` represents the aperture.

# Arguments
- `Nx`: length along x direction
- `Ny`: length along y direction
- `w`: half width of each side or (radius, number of side) for regular polygon
- `shape`: shape of the aperture. `:circ`, `:rect` and `:regular_polygon`
- `origin`: origin of aperture
- `rim`: edge width
- `T`: data type of mask matrix

"""
function aperture(
    Nx::Integer,
    Ny::Integer,
    w;
    shape=:circ,
    origin=middle.((Nx, Ny)),
    rim=RIM_WIDTH,
    T=PRECISION
)
    x0, y0 = origin
    rim = rim / RIM_CONSTANT
    x = (1:Nx) .- x0
    y = (1:Ny) .- y0
    yt = transpose(y)
    w = w .* (1, 1)
    ap = Matrix{T}(undef, Nx, Ny)
    if shape == :circ
        r = @. sqrt(x^2 + yt^2)
        t = @. yt / x
        @. t[isnan(t)] = 1
        R = @. sqrt((1 + t^2) / (1 / w[1]^2 + t^2 / w[2]^2))
        @. R[isinf(t)] = w[2]
        c = @. (R - r) / rim
        @. c[isnan(c)] = 1
        @. ap = (1 + tanh(c)) / 2
    elseif shape == :rect
        @. ap = (1 + tanh((w[1] - abs(x)) / rim)) *
                (1 + tanh((w[2] - abs(yt)) / rim)) / 4
    elseif shape == :regular_polygon
        α = 2π / w[2]
        a = w[1] * cos(α / 2)
        β = @. atan(-yt / abs(x)) + π / 2
        @. β[isnan(β)] = 0
        γ = @. abs(β % α - α / 2)
        d = @. a / cos(γ)
        r = @. sqrt(x^2 + yt^2)
        c = @. (d - r) / rim
        @. c[isnan(c)] = 1
        @. ap = (1 + tanh(c)) / 2
    else
        error("no such shape")
    end
    ap
end

function aperture(
    N::Integer,
    w::Real;
    origin=middle(N),
    rim=RIM_WIDTH,
    T=PRECISION
)
    x0 = origin
    rim = rim / RIM_CONSTANT
    x = (1:N) .- x0
    ap = Vector{T}(undef, N)
    @. ap = (1 + tanh((w - abs(x)) / rim)) / 2
    ap
end

"""
    aperture(Nx::Integer, Ny::Integer; img_path::String, img_size::NTuple{Integer, 2}, T=$PRECISION)

Return a `Nx` \$\\times\$ `Ny` mask matrix of eltype `T` represents the aperture.

# Arguments
- `Nx`: length along x direction
- `Ny`: length along y direction
- `img_path`: path of img file
- `img_size`: 
- `T`: data type of mask matrix

"""
function aperture(
    Nx::Integer, 
    Ny::Integer; 
    img_path::String, 
    img_size::NTuple{2, Integer}, 
    T=PRECISION
)
    img_source = Gray.(load(img_path))
    img_permute = img2mat(img_source)
    img_resize = imresize(img_permute, img_size) |> Matrix{T}
    ap_size = (Nx, Ny)
    offset = @. (ap_size - img_size) ÷ 2 + 1
    ap = PaddedView(0, img_resize, ap_size, offset) |> Matrix{T}
    ap
end