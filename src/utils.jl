middle(N::Integer) = N ÷ 2 + 1
middle(v::AbstractVector) = v[middle(length(v))]

img2mat(img::AbstractMatrix) = rotr90(img) # reverse(permutedims(img, (2, 1)), dims=2)
mat2img(mat::AbstractMatrix) = rotl90(mat) # reverse(permutedims(mat, (2, 1)), dims=1)

chirp(v::AbstractVector) = v .^ 2
chirp(v::NTuple{2,AbstractVector}) = chirp(v[1]) .+ transpose(chirp(v[2]))