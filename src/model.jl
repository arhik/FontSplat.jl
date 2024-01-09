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
	colors
	scales
	opacities
end


using Rotations

function genSplatData(n)
	means = rand(2, n)
	rots = rand(4, n)
	colors = rand(3, n)
	scales = rand(2, n)
	opacities = rand(1, n)
	return GSplatData(means, rots, colors, scales, opacities) 
end

splatData = genSplatData(25)
splats = splatData
idx = 1

gt = load(joinpath(
	ENV["HOME"],
	"Desktop",
	"six.png"
))

img = ones(RGBA{N0f8}, (400, 400))
cimgview = channelview(img)
cgtview = channelview(gt)

function drawSplat!(cov2d, quadView, point, color, opacity, TPrev)
	sz = size(quadView)
	idxs = CartesianIndices(Tuple([1:sz[1], 1:sz[2]])) |> collect
	αs = ones(N0f8, sz...)
	invCov2d = inv(cov2d)
	for idx in idxs
		delta = [idx.I[1] - point[1], idx.I[2] - point[2]]./[sz...]
		dist = (delta |> adjoint)*invCov2d*delta
		α = opacity*exp(-dist)
		αs[idx] = α |> N0f8
	end
	T = TPrev.*((1.0 |> N0f8) .- αPrev)
	quadView[:, :] .+= colorview(
		RGBA, 
		repeat([color[1]] .|> N0f8, inner=sz).*αs.*T, 
		repeat([color[2]] .|> N0f8, inner=sz).*αs.*T, 
		repeat([color[3]] .|> N0f8, inner=sz).*αs.*T, 
		αs
	)
	TPrev .= T
end

function renderSplats(splats, cimage)
	image = colorview(RGBA, cimage)
	transmittance = ones(N0f8, size(image))
	nPoints = splats.means |> size |> last
	#forward
	for idx in 1:nPoints-1
		point = [size(cimage)[2:3]...].*splats.means[:, idx]
		rot = reshape(splats.rotations[:, idx], (2, 2))
		scale = Diagonal(splats.scales[:, idx])
		color = splats.colors[:, idx]
		W = rot*scale

		cov2d = W*adjoint(W)
		Δ = det(cov2d)

		if Δ < 0.0
			continue
		end
		
		λs = eigen(cov2d)
		r = ceil(3.0*sqrt(maximum(λs.values)))
		bb = [size(cimage)[2:3]...].*[-1 1; -1 1]*r .+ point
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], size(image, 2));
			max(1, bb[2][1]) min(bb[2, 2], size(image, 1))
		]) .|> Int
		# quad = zeros(RGBA{N0f8}, bb[1, :] | length, b[2, :] |> length)
		quadView = view(image, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		opacity = splats.opacities[idx]
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		drawSplat!(cov2d, quadView, point, color, opacity, TPrev)
	end
	return image
	# backward
	for idx in nPoints:1
		point = [size(cimage)[2:3]...].*splats.means[:, idx]
		rot = reshape(splats.rotations[:, idx], (2, 2))
		scale = Diagonal(splats.scales[:, idx])
		color = splats.colors[:, idx]
		W = rot*scale

		cov2d = W*adjoint(W)
		Δ = det(cov2d)

		if Δ < 0.0
			continue
		end
		
		λs = eigen(cov2d)
		r = ceil(3.0*sqrt(maximum(λs.values)))
		bb = [size(cimage)[2:3]...].*[-1 1; -1 1]*r .+ point
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], size(image, 2));
			max(1, bb[2][1]) min(bb[2, 2], size(image, 1))
		]) .|> Int
		# quad = zeros(RGBA{N0f8}, bb[1, :] | length, b[2, :] |> length)
		quadView = view(image, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		opacity = splats.opacities[idx]
		sz = size(quadView)
		idxs = CartesianIndices(Tuple([1:sz[1], 1:sz[2]])) |> collect
		αs = ones(N0f8, sz...)
		invCov2d = inv(cov2d)
		for idx in idxs
			delta = [idx.I[1] - point[1], idx.I[2] - point[2]]./[sz...]
			dist = (delta |> adjoint)*invCov2d*delta
			α = opacity*exp(-dist)
			αs[idx] = α |> N0f8
		end
		# calculate gradients for colors of splats
		Δc = αs.*TPrev
		# calculate Transmittance online
		TPrev .= TPrev.*n0f8.(clamp.(1.0./(1.0 .- αs), 0.0, 1.0))
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
