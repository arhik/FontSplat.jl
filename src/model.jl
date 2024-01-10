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
		RGBA, 
		repeat([color[1]] .|> N0f8, inner=sz).*αs.*T, 
		repeat([color[2]] .|> N0f8, inner=sz).*αs.*T, 
		repeat([color[3]] .|> N0f8, inner=sz).*αs.*T, 
		αs
	)
	TPrev .= T
	αPrev .= αs
end

function renderSplats(splats, cimage)
	image = colorview(RGBA, cimage)
	transmittance = ones(N0f8, size(image))
	nPoints = splats.means |> size |> last
	alpha = zeros(N0f8, size(image))
	#forward
	for idx in 1:nPoints
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
			max(1, bb[1][1]) min(bb[1, 2], size(image, 2));
			max(1, bb[2][1]) min(bb[2, 2], size(image, 1))
		]) .|> Int
		# quad = zeros(RGBA{N0f8}, bb[1, :] | length, b[2, :] |> length)
		quadView = view(image, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		αPrev = view(alpha, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		opacity = splats.opacities[idx]
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		drawSplat!(cov2d, quadView, point, color, opacity, TPrev, αPrev)
	end
		# backward
	S = zeros(eltype(image), size(image))
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
		Δo = zeros(Float32, size(quadView))
		Δσ = zeros(Float32, size(quadView))
		Δμ = zeros(Float32, 2, sz...)
		ΔΣ = zeros(Float32, 2, 2, sz...)
		for pidx in idxs
			delta = [pidx.I[1] - point[1], pidx.I[2] - point[2]]./[sz...]
			dist = (delta |> adjoint)*invCov2d*delta
			Δμ[:, pidx] .= invCov2d*delta
			ΔΣ[:, :, pidx] .= -0.5.*(invCov2d*delta*(delta |> adjoint)*(invCov2d |> adjoint))
			α = opacity*exp(-dist)
			αs[pidx] = α |> N0f8
			Δo[pidx] = exp(-dist)
			Δσ[pidx] = -opacity*exp(-dist)
		end
		Δo = permuteddimsview(repeat(Δo, 1, 1, 4), (3, 1, 2))
		Δσ = permuteddimsview(repeat(Δσ, 1, 1, 4), (3, 1, 2))
		# calculate gradients for colors of splats
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		Δc = permuteddimsview(repeat(αs.*TPrev, 1, 1, 4), (3, 1, 2))
		# calculate gradients for α of splats
		SPrev = view(S, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		Δα = (colorview(
				RGBA, 
				repeat([color[1]] .|> N0f8, inner=sz).*TPrev, 
				repeat([color[2]] .|> N0f8, inner=sz).*TPrev, 
				repeat([color[3]] .|> N0f8, inner=sz).*TPrev, 
				ones(N0f8, sz)
			) |> channelview) .- ((SPrev.*n0f8.(clamp.(1.0./(1.0 .- αs), 0.0, 1.0))) |> channelview)

		grad = (ΔC) -> begin
			cGrad = sum(Δc.*view(ΔC, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...)), dims=(2, 3))[:][1:3]
			#αGrad = sum(Δα.*view(ΔC, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...)))
			αGrad = Δα.*Δc.*view(ΔC, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
			oGrad = sum(Δo.*αGrad)
			σGrad = sum(Δσ.*αGrad)
			#μGrad = map((_idx) -> Δσ[_idx]*grads1, 1:size()
			μGrad = sum(Δμ.*σGrad, dims=(2, 3))[:]
			ΣGrad = sum(ΔΣ.*σGrad, dims=(3, 4))[:, :]
			return (cGrad, oGrad, μGrad, ΣGrad)
		end
		push!(grads, grad)
		# update S
		SPrev .+= colorview(
			RGBA, 
			repeat([color[1]] .|> N0f8, inner=sz).*αs.*TPrev, 
			repeat([color[2]] .|> N0f8, inner=sz).*αs.*TPrev, 
			repeat([color[3]] .|> N0f8, inner=sz).*αs.*TPrev, 
			ones(N0f8, sz)
		)
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
	return sum(((1 .-img) .- gt).^2)/(2.0*length(img))
end

function errorGrad(img, gt)
	cimgview = channelview(img)
	gtview = channelview(gt)
	s = error(cimgview, gtview)
	@info "loss : " s
	return -(1.0 .-cimgview .- gtview)/prod(size(cimgview))
end

n = 10

staticRot = rand(4, n);
staticMeans = rand(2, n)
#staticOpacities = ones(1, n)
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
img = ones(RGBA{N0f8}, imgSize)
cimgview = channelview(img)

(outimage, grads) = renderSplats(splatDataOriginal, cimgview)
save("fontsplat.jpg", outimage)


gt = load("fontsplat.jpg")
gt = colorview(RGBA, gt, ones(N0f8, size(gt)))
cgtview = channelview(gt)

splatData = genSplatData(n)
(outimage, grads) = renderSplats(splatData, cimgview)

lr = -0.011
for i in 1:1000
	@info "iteration: $(i)"
	img = ones(RGBA{N0f8}, imgSize)
	cimgview = channelview(img)
	(outimage, grads) = renderSplats(splatData, cimgview)
	println("saving fontsplat $(i)")
	save("fontsplat$(i).jpg", outimage)
	ΔC = errorGrad(outimage, gt)
	nPoints = size(splatData.means, 2)
	for idx in nPoints:-1:1
		(cGrad, oGrad, μGrad, ΣGrad) = grads[idx](ΔC)
		#splatData.means[:, idx] .+= lr.*μGrad
		splatData.colors[:, idx] .+= lr.*cGrad
		splatData.opacities[:, idx] .+= lr.*oGrad
	end
end


