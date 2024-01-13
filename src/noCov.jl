#using Pkg
#Pkg.activate("FontSplat")
#using Flux
using Images
#using ImageView
using LinearAlgebra
using ImageInTerminal
using Statistics
using ImageView

struct GSplatData
	means
	rotations
	colors
	scales
	opacities
end

using Infiltrator

using Rotations

function drawSplat!(cov2d, quadView, point, color, opacity, TPrev, αPrev)
	sz = size(quadView)[2:3]
	idxs = CartesianIndices(Tuple([1:sz[1], 1:sz[2]]))
	αs = zeros(sz...)
	invCov2d = inv(cov2d)
	for idx in idxs
		delta = [idx.I[1] - point[1], idx.I[2] - point[2]]./[sz...]
		dist = (delta |> adjoint)*invCov2d*delta
		α = opacity*exp(-dist)
		αs[idx] = α
	end
	T = TPrev.*(1.0 .- αPrev)
	quadView .+= repeat(color, 1, sz...).*reshape(αs.*T, (1, sz...))
	TPrev .= T
	αPrev .= αs
end

function renderSplats(splats, cimage)
	sz = size(cimage)[2:3]
	transmittance = ones(sz)
	nPoints = splats.means |> size |> last
	alpha = zeros(size(cimage)[2:3])
	#forward
	for idx in 1:nPoints
		point = [sz...].*(splats.means[:, idx])
		rot = reshape(splats.rotations[:, idx], (2, 2))
		scale = (Diagonal(splats.scales[:, idx]))
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
		bb = [sz...].*[-1 1; -1 1]*r .+ point/2.0
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], size(cimage, 2));
			max(1, bb[2][1]) min(bb[2, 2], size(cimage, 3))
		]) .|> Int
		quadView = view(cimage, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		αPrev = view(alpha, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		opacity = splats.opacities[idx]
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		drawSplat!(cov2d, quadView, point, color, opacity, TPrev, αPrev)
	end
	# backward
	S = zeros(Float32, size(cimage)...)
	grads = []
	#cimage = deepcopy(cimage)
	for idx in nPoints:-1:1
		point = [sz...].*splats.means[:, idx]
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
		bb = [sz...].*[-1 1; -1 1]*r .+ point./2.0
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], sz[1]);
			max(1, bb[2][1]) min(bb[2, 2], sz[2])
		]) .|> Int
		quadView = view(cimage, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		opacity = splats.opacities[idx]
		quadSize = size(quadView)[2:3]
		idxs = CartesianIndices(Tuple([1:quadSize[1], 1:quadSize[2]])) |> collect
		αs = ones(quadSize...)
		invCov2d = inv(cov2d)
		Δo = zeros(Float32, 1, quadSize...)
		Δσ = zeros(Float32, 1, quadSize...)
		Δμ = zeros(Float32, 2, quadSize...)
		ΔΣ = zeros(Float32, 2, 2, quadSize...)
		for pidx in idxs
			delta = [pidx.I[1] - point[1], pidx.I[2] - point[2]]./[sz...]
			dist = (delta |> adjoint)*invCov2d*delta
			Δμ[:, pidx] .= invCov2d*delta./[sz[1], sz[2]]
			ΔΣ[:, :, pidx] .= -0.5.*(invCov2d*delta*(delta |> adjoint)*(invCov2d |> adjoint))
			α = opacity*exp(-dist)
			αs[pidx] = α
			Δo[:, pidx] .= exp(-dist)
			Δσ[:, pidx] .= -opacity*exp(-dist)
		end
		# calculate gradients for colors of splats
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		Δc = reshape(αs.*TPrev, 1, quadSize...)
		# calculate gradients for α of splats
		SPrev = view(S, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		Δα = repeat(color, 1, quadSize...).*reshape(TPrev, 1, quadSize...) .- SPrev.*(1.0./(1.0 .- reshape(αs, 1, size(αs)...)))
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
		SPrev .+= repeat(color, 1, quadSize...).*reshape(αs.*TPrev, 1, quadSize...)
		# update Transmittance online
		TPrev .= TPrev.*(1.0./(1.0 .- αs))
	end
	return (cimage, grads)
end

function forward(splatData, cimgview, gt)
	outimage = renderSplats(splatData, cimgview)
	return error(outimage, gt)
end

function error(img, gt)
	return sum((img .- gt).^2)/(2.0*length(img))
end

function errorGrad(img, gt)
	cimgview = channelview(img)
	gtview = channelview(gt)
	s = error(cimgview, gtview)
	@info "loss : " s
	return (cimgview .- gtview)/length(cimgview)
end

n = 10

staticRot = repeat([1, 0, 0, 1], 1, n);
staticMeans = repeat([0.5, 0.5], 1, n)
staticOpacities = rand(1, n)
staticScales = 0.5.*ones(2, n)
staticColors = rand(3, n)# ([1.0, 0.0, 0.0], 1, n)

function genSplatData(n)
	means = rand(2, n)
	rots = staticRot#rand(4, n) 
	colors = rand(3, n)
	scales = staticScales # rand(2, n)
	opacities = rand(1, n)
	return GSplatData(means, rots, colors, scales, opacities) 
end

splatDataOriginal = genSplatData(n)

imgSize = (20, 30)

imgsrc = nothing#"pointgraphics.jpg"

if imgsrc == nothing
	img = zeros(RGB{N0f8}, imgSize)
	cimgview = channelview(img) .|> float
	(outimage, grads) = renderSplats(splatDataOriginal, cimgview)
	save("fontsplat.jpg", colorview(RGB{N0f8}, n0f8.(outimage)))
else
	img = load(imgsrc)
	img = imresize(img, imgSize)
	save("fontsplat.jpg", img)
end

gt = load("fontsplat.jpg")

splatData = genSplatData(n)
lr = 0.01
for i in 0001:20000
	@info "iteration: $(i)"
	target = zeros(RGB{N0f8}, imgSize)
	targetview = channelview(target) .|> float
	(tmpimage, grads) = renderSplats(splatData, targetview)
	println("saving fontsplat $(i)")
	if i%100 == 0
		save("fontsplat$(i).jpg", colorview(RGB{N0f8}, n0f8.(tmpimage)))
	end
	ΔC = errorGrad(tmpimage, gt)
	nPoints = size(splatData.means, 2)
	
	"""
	for pb in grads
		fs = propertynames(pb) .|> string
		bbfield = nothing
		for f in fs
			if endswith(f, "bb")
				bbfield = f
			end
		end
		bbox = getfield(pb, Symbol(bbfield))
		if bbox.contents != Core.Box([1 32; 1 32]).contents
			@infiltrate
		end
	end
	"""
	
	for idx in nPoints:-1:1
		(cGrad, oGrad, μGrad, ΣGrad) = grads[idx](ΔC)
		splatData.means[:, idx] .-= (lr.*μGrad)
		splatData.colors[:, idx] .-= lr.*cGrad
		splatData.opacities[:, idx] .-= lr.*oGrad
		#if any(abs.(splatData.means) .> 1.0)
		#	@warn "means are diverging"
		#end
	end
end

# THESE ARE TESTS

# Testing Color Gradients ...

function testColorsGrads()
	n = 4
	αs = rand(n);
	cns = rand(3, n);
	ci = zeros(3);

	t = 1.0
	for idx in 1:n
		ci += cns[:, idx].*αs[idx].*t
		t *= (1 - αs[idx])
	end

	cnhats = rand(3, n)

	for itr in 1:100000
		local t
		cihat = zeros(3)
		t = 1.0
		for idx in 1:n
			cihat += cnhats[:, idx].*αs[idx].*t
			t *= (1 - αs[idx])
		end
		Δci = cihat .- ci
		αsPrev = 0.0
		for idx in n:-1:1
			α = αs[idx]
			t = t/(1.0 - αsPrev)
			Δcn = α.*t.*Δci
			αsPrev = α
			cnhats[:, idx] .-= 0.001.*Δcn
		end
		@info "Loss" sum(Δci.^2)
		@info "cn" sum((cnhats .- cns).^2)
	end
	return (cns, cnhats)
end
