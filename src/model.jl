#using Pkg
#Pkg.activate("FontSplat")
#using Flux
using Images
#using ImageView
using LinearAlgebra
using ImageInTerminal
using Statistics

struct GSplatData
	means
	rotations
	colors
	scales
	opacities
end

using Rotations

function drawSplat!(cov2d, quadView, point, color, opacity, TPrev, αPrev)
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
		RGB, 
		(clamp(color[1], 0.0, 1.0) .|> N0f8).*αs.*T, 
		(clamp(color[2], 0.0, 1.0) .|> N0f8).*αs.*T, 
		(clamp(color[3], 0.0, 1.0) .|> N0f8).*αs.*T, 
	)
	TPrev .= T
	αPrev .= αs
end

function renderSplats(splats, cimage)
	image = colorview(RGB, cimage)
	transmittance = ones(N0f8, size(image))
	nPoints = splats.means |> size |> last
	alpha = zeros(N0f8, size(image))
	#forward
	for idx in 1:nPoints
		point = [size(cimage)[2:3]...].*splats.means[:, idx]
		rot = reshape(splats.rotations[:, idx], (2, 2))
		scale = exp.(Diagonal(splats.scales[:, idx]))
		color = splats.colors[:, idx]
		W = rot*scale

		cov2d = W*adjoint(W)
		Δ = det(cov2d)

		if Δ < 0.0
			@warn "Determinant is negative"
			continue
		end
		
		λs = eigen(cov2d)
		r = ceil(3.0*sqrt(maximum(λs.values)))
		bb = [size(cimage)[2:3]...].*[-1 1; -1 1]*r .+ point
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], size(image, 1));
			max(1, bb[2][1]) min(bb[2, 2], size(image, 2))
		]) .|> Int
		# quad = zeros(RGB{N0f8}, bb[1, :] | length, b[2, :] |> length)
		quadView = view(image, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		αPrev = view(alpha, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		opacity = splats.opacities[idx]
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		drawSplat!(cov2d, quadView, point, color, opacity, TPrev, αPrev)
	end
		# backward
	S = zeros(Float32, 3, size(image)...)
	grads = []
	image = deepcopy(image)
	for idx in nPoints:-1:1
		point = [size(cimage)[2:3]...].*splats.means[:, idx]
		rot = reshape(splats.rotations[:, idx], (2, 2))
		scale = Diagonal(splats.scales[:, idx])
		color = splats.colors[:, idx]
		W = rot*scale

		cov2d = W*adjoint(W)
		Δ = det(cov2d)

		if Δ < 0.0
			@warn "Determinant is negative"
			continue
		end
		
		λs = eigen(cov2d)
		r = ceil(3.0*sqrt(maximum(λs.values)))
		bb = [size(cimage)[2:3]...].*[-1 1; -1 1]*r .+ point
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], size(image, 1));
			max(1, bb[2][1]) min(bb[2, 2], size(image, 2))
		]) .|> Int
		quadView = view(image, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		opacity = splats.opacities[idx]
		sz = size(quadView)
		idxs = CartesianIndices(Tuple([1:sz[1], 1:sz[2]])) |> collect
		αs = ones(N0f8, sz...)
		invCov2d = inv(cov2d)
		Δo = 0.0
		Δσ = 0.0
		Δμ = zeros(Float32, 2, sz...)
		ΔΣ = zeros(Float32, 2, 2, sz...)
		for pidx in idxs
			delta = [pidx.I[1] - point[1], pidx.I[2] - point[2]]./[sz...]
			dist = (delta |> adjoint)*invCov2d*delta
			Δμ[:, pidx] .= invCov2d*delta
			ΔΣ[:, :, pidx] .= -0.5.*(invCov2d*delta*(delta |> adjoint)*(invCov2d |> adjoint))
			α = opacity*exp(-dist)
			αs[pidx] = α |> N0f8
			Δo += exp(-dist)
			Δσ += -opacity*exp(-dist)
		end
		Δo = Δo/length(quadView)
		Δσ = Δσ/length(quadView)
		# calculate gradients for colors of splats
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		Δc = permuteddimsview(repeat(αs.*TPrev, 1, 1, 3), (3, 1, 2))
		# calculate gradients for α of splats
		SPrev = view(S, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		Δα = colorview(
			RGB,
			repeat([color[1]], sz...).*TPrev .- (SPrev[1, :, :].*n0f8.(clamp.(1.0./(1.0 .- αs), 0.0, 1.0))), 
			repeat([color[2]], sz...).*TPrev .- (SPrev[2, :, :].*n0f8.(clamp.(1.0./(1.0 .- αs), 0.0, 1.0))), 
			repeat([color[3]], sz...).*TPrev .- (SPrev[3, :, :].*n0f8.(clamp.(1.0./(1.0 .- αs), 0.0, 1.0))), 
		) |> channelview
		grad = (ΔC) -> begin
			cGrad = Δc.*view(ΔC, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
			cgGrad = sum(cGrad, dims=(2, 3))[:]
			αGrad = sum(Δα.*cGrad, dims=1)
			oGrad = sum(Δo.*αGrad)
			σGrad = sum(Δσ.*αGrad)
			μGrad = sum(Δμ.*σGrad, dims=(2, 3))[:]
			ΣGrad = sum(ΔΣ.*σGrad, dims=(3, 4))[:, :]
			return (cgGrad, oGrad, μGrad, ΣGrad)
		end
		push!(grads, grad)
		# update S
		SPrev .+= colorview(
			RGB, 
			repeat([color[1]] .|> N0f8, sz...).*αs.*TPrev, 
			repeat([color[2]] .|> N0f8, sz...).*αs.*TPrev, 
			repeat([color[3]] .|> N0f8, sz...).*αs.*TPrev, 
		) |> channelview
		# update Transmittance online
		TPrev .= TPrev.*n0f8.(clamp.(1.0./(1.0 .- αs), 0.0, 1.0))
	end
	return (image, grads)
end

function forward(splatData, cimgview, gt)
	outimage = renderSplats(splatData, cimgview)
	return error(outimage, gt)
end

function error(img, gt)
	return sum(((img) .- gt).^2)/(2.0*length(img))
end

function errorGrad(img, gt)
	cimgview = channelview(img)
	gtview = channelview(gt)
	s = error(cimgview, gtview)
	@info "loss : " s
	return (cimgview .- gtview)/length(cimgview)
end

n = 10

staticRot = rand(4, n);
staticMeans = rand(2, n)
staticOpacities = rand(1, n)
staticScales = ones(2, n)
staticColors = rand(3, n)

function genSplatData(n)
	means = staticMeans
	rots = staticRot#rand(4, n) 
	colors = rand(3, n)
	scales = staticScales # rand(4, n)
	opacities = rand(1, n)
	return GSplatData(means, rots, colors, scales, opacities) 
end

splatDataOriginal = genSplatData(n)

imgSize = (64, 64)

img = ones(RGB{N0f8}, imgSize)
cimgview = channelview(img)
(outimage, grads) = renderSplats(splatDataOriginal, cimgview)
save("fontsplat.jpg", outimage)

gt = load("fontsplat.jpg")
gt = colorview(RGB{N0f8}, gt)
splatData = genSplatData(n)
lr = 0.01
for i in 1:1000
	@info "iteration: $(i)"
	img = ones(RGB{N0f8}, imgSize)
	cimgview = channelview(img)
	(outimage, grads) = renderSplats(splatData, cimgview)
	println("saving fontsplat $(i)")
	save("fontsplat$(i).jpg", outimage)
	ΔC = errorGrad(outimage, gt)
	nPoints = size(splatData.means, 2)
	for idx in nPoints:-1:1
		(cGrad, oGrad, μGrad, ΣGrad) = grads[idx](ΔC)
		splatData.means[:, idx] .-= lr.*μGrad
		splatData.colors[:, idx] .-= lr.*cGrad
		splatData.opacities[:, idx] .-= lr.*oGrad
		if any(abs.(splatData.means) .> 1.0)
			@warn "means are diverging"
		end
	end
end

