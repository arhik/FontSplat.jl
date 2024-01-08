#using Pkg
#Pkg.activate("FontSplat")
#using Flux
using Images
#using ImageView
using LinearAlgebra
using ImageInTerminal


struct GSplatData
	means
	rotations
	scales
	opacities
end

n = 10

using Rotations

function genSplatData()
	means = rand(2, n)
	rots = rand(4, n)
	scales = rand(2, n)
	opacities = rand(1, n)
	return GSplatData(means, rots, scales, opacities) 
end

splatData = genSplatData()
splats = splatData
idx = 1

gt = load(joinpath(
	ENV["HOME"],
	"Desktop",
	"six.png"
))

img = ones(RGBA{N0f8}, size(gt))
cimgview = channelview(img)
cgtview = channelview(gt)

function drawSplat!(cov2d, quadView, opacity, TPrev, αPrev)
	sz = size(quadView)
	idxs = CartesianIndices(Tuple([1:sz[1], 1:sz[2]])) |> collect
	αs = ones(N0f8, sz...)
	invCov2d = inv(cov2d)
	for idx in idxs
		delta = [idx.I[1] - sz[1]/2, idx.I[2] - sz[2]]./[sz...]
		dist = (delta |> adjoint)*invCov2d*delta
		α = opacity .* (exp(-dist))
		αs[idx] = α |> N0f8
	end
	T = TPrev.*((1.0 |> N0f8) .- αPrev)
	quadView[:, :] .+= colorview(
		RGBA, 
		red.(quadView).*αs.*T, 
		green.(quadView).*αs.*T, 
		blue.(quadView).*αs.*T, 
		αs
	)
	αPrev .= αs
	TPrev .= T
end


function renderSplats(splats, cimage)
	image = colorview(RGBA, cimage)
	transmittance = ones(N0f8, size(image))
	nPoints = splats.means |> size |> last
	for idx in 1:nPoints
		point = [size(cimage)[2:3]...].*splats.means[:, idx]
		rot = reshape(splats.rotations[:, idx], (2, 2))
		scale = Diagonal(splats.scales[:, idx])
		W = rot*scale

		cov2d = W*adjoint(W)
		Δ = det(cov2d)

		if Δ < 0.0
			continue
		end
		
		λs = eigen(cov2d)
		r = maximum(λs.values)
		bb = [size(cimage)[2:3]...].*[-1 1; -1 1]*r .+ point
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], size(image, 1));
			max(1, bb[2][1]) min(bb[2, 2], size(image, 2))
		]) .|> Int
		quadView = view(image, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		αPrev = zeros(N0f8, size(quadView))
		opacity = splats.opacities[idx]
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		drawSplat!(cov2d, quadView, opacity, TPrev, αPrev)
	end
	
	return image
end


outimage = renderSplats(splatData, cimgview)

function forward(splatData, cimgview, gt)
	outimage = renderSplats(splatData, cimgview)
	return error(outimage, gt)
end


function error(img, gt)
	return sum(((1 .-img) .- gt).^2)/(2.0*length(img))
end

function errorGrad(img, gt)
	s = error(img, gt)
	return -(1.0 .-img .- gt)/prod(size(img))
end

save("fontsplat.jpg", outimage)
