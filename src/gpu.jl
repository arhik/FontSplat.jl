using CUDA
using Flux

a = rand(Float32, 32, 32, 3) |> gpu

function drawSplat!(cov2d)
    sz = size(quadView)[1:2]
    idxs = CartesianIndices(Tuple([1:sz[1], 1:sz[2]]))
    αs = zeros(sz...)
    invCov2d = inv(cov2d)
    for idx in idxs
        delta = [[idx.I[1] - point[1], idx.I[2] - point[2]]./[sz...]
        dist = (delta |> adjoint)*invCov2d*delta
        α = opacity*exp(-dist)
        αs[idx] = α
    end
    quadView .+= repeat(reshape(color, 1, 1, 3), sz...).*reshape(αs.*TPrev, (sz..., 1))
    TPrev .*= (1.0f0 .- αPrev)
    αPrev .= αs
end
