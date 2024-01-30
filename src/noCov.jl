# using Pkg
# Pkg.activate("FontSplat")
# using Flux
using Images
# using ImageView
using LinearAlgebra
using ImageInTerminal
using Statistics
using ImageView
using Infiltrator

include("functions.jl")


struct GSplatData
	means
	rotations
	colors
	scales
	opacities
end

function Base.show(io::IO, g::GSplatData)
	println("GSplatData")
	for i in propertynames(g)
		println("\t $(i) :$(getfield(g, i))")
	end
end

using Infiltrator

using Rotations

function drawSplat!(cov2d, quadView, point, color, opacity, TPrev, αPrev)
	sz = size(quadView)[1:2]
	idxs = CartesianIndices(Tuple([1:sz[1], 1:sz[2]]))
	αs = zeros(sz...)
	invCov2d = inv(cov2d)
	for idx in idxs
		delta = [idx.I[1] - point[1], idx.I[2] - point[2]]./[sz...]
		dist = (delta |> adjoint)*invCov2d*delta
		α = opacity*exp(-dist)
		αs[idx] = α
	end
	quadView .+= repeat(reshape(color, 1, 1, 3), sz...).*reshape(αs.*TPrev, (sz..., 1))
	TPrev .*= (1.0 .- αPrev)
	αPrev .= αs
end

function sigmoid(x)
	z = exp.(x)
	return z./(1 .+ z)
end

function renderSplats(splats, cimage)
	sz = size(cimage)[1:2]
	transmittance = ones(sz)
	nPoints = splats.means |> size |> last
	alpha = zeros(sz)
	#forward
	for idx in 1:nPoints
		point = [sz...].*((splats.means[:, idx]) .|> sigmoid)
		
		# Constructing 2D rotation matrices
		rot = RotZ((pi).*(splats.rotations[:, idx])...)[1:2, 1:2]
		
		scale = (Diagonal(clamp.(splats.scales[:, idx] .|> exp, 0.0, 1.0)))
		color = splats.colors[:, idx]
		W = rot*scale
		
		cov2d = W*adjoint(W)
		cov2d[1, 1] += 0.05
		cov2d[2, 2] += 0.05
		
		Δ = det(cov2d)
		
		if Δ == 0.0
			@warn "Determinant is negative"
			continue
		end
		
		λs = eigen(cov2d)
		r = ceil(3.0*sqrt(maximum(λs.values)))
		bb = [sz...].*[-1 1; -1 1]*r .+ point/2.0
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], size(cimage, 1));
			max(1, bb[2][1]) min(bb[2, 2], size(cimage, 2))
		]) .|> Int
		quadView = view(cimage, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...), :)
		αPrev = view(alpha, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		opacity = splats.opacities[idx]
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		drawSplat!(cov2d, quadView, point, color, opacity, TPrev, αPrev)
	end
	# backward
	S = zeros(Float32, size(cimage)...)
	alpha .= 0.0
	local grads
	grads = []
	#cimage = deepcopy(cimage)
	for idx in nPoints:-1:1
		point = [sz...].*((splats.means[:, idx]) .|> sigmoid)
		rot = RotZ((pi/2.0).*(splats.rotations[:, idx])...)[1:2, 1:2]
		scale = Diagonal(clamp.(splats.scales[:, idx] .|> exp, 0.0, 1.0))
		color = splats.colors[:, idx]
		W = rot*scale
		
		cov2d = W*adjoint(W)
		cov2d[1, 1] += 0.05
		cov2d[2, 2] += 0.05
		Δ = det(cov2d)
		
		if Δ == 0.0
			@warn "Determinant is negative"
			push!(grads, (_) -> (0.0, 0.0, 0.0, 0.0, Diagonal{Float32}(undef, 2).=0.0))
			continue
		end
		
		λs = eigen(cov2d)
		r = ceil(3.0*sqrt(maximum(λs.values)))
		bb = [sz...].*[-1 1; -1 1]*r .+ point/2.0
		bb = ceil.([
			max(1, bb[1][1]) min(bb[1, 2], sz[1]);
			max(1, bb[2][1]) min(bb[2, 2], sz[2])
		]) .|> Int
		quadView = view(cimage, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...), :)
		opacity = splats.opacities[idx]
		quadSize = size(quadView)[1:2]
		idxs = CartesianIndices(Tuple([1:quadSize[1], 1:quadSize[2]])) |> collect
		αs = zeros(quadSize)
		invCov2d = inv(cov2d)
		Δo = zeros(Float32, quadSize..., 1)
		Δσ = zeros(Float32, quadSize..., 1)
		Δμ = zeros(Float32, quadSize..., 2)
		ΔΣ = zeros(Float32, quadSize..., 2, 2)
		for pidx in idxs
			delta = [pidx.I[1] - point[1], pidx.I[2] - point[2]]./[sz...]
			dist = (delta |> adjoint)*invCov2d*delta
			Δμ[pidx, :] .= invCov2d*delta./[sz[1], sz[2]]
			ΔΣ[pidx, :, :] .= -0.5.*(invCov2d*delta*(delta |> adjoint)*(invCov2d |> adjoint))
			α = opacity*exp(-dist)
			αs[pidx] = α
			Δo[pidx, :] .= exp(-dist)
			Δσ[pidx, :] .= -opacity*exp(-dist)
		end
		# calculate gradients for colors of splats
		TPrev = view(transmittance, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		αPrev = view(alpha, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		TPrev .= TPrev.*(1.0./(1.0 .- αPrev))
		αPrev .= αs
		Δc = reshape(αs.*TPrev, quadSize..., 1)
		# calculate gradients for α of splats
		SPrev = view(S, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...), :)
		Δα = repeat(reshape(color, 1, 1, 3), quadSize..., 1).*reshape(TPrev, quadSize..., 1) .- SPrev.*(1.0./(1.0 .- reshape(αs, size(αs)..., 1)))
		grad = (ΔC) -> begin
			cGrad = Δc.*view(ΔC, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...), :)
			cgGrad = sum(cGrad, dims=(1, 2))[:]
			αGrad = sum(Δα.*cGrad, dims=3)
			oGrad = sum(Δo.*αGrad)
			σGrad = sum(Δσ.*αGrad)
			μGrad = sum(Δμ.*σGrad, dims=(1, 2))[:]
			ΣGrad = sum(ΔΣ.*σGrad, dims=(1, 2))[1, 1, :, :]
			wGrad = ΣGrad*W + (ΣGrad |> adjoint)*W
			rGrad = wGrad*(scale |> adjoint)
			RGrad = [0 -1; 1 0]*rot*rGrad
			θGrad = atan(RGrad[2, 1], RGrad[2, 2])
			sGrad = (rot |> adjoint)*wGrad
			return (cgGrad, oGrad, μGrad, θGrad, sGrad)
		end
		push!(grads, grad)
		# update S
		SPrev .+= repeat(reshape(color, 1, 1, 3), quadSize..., 1).*reshape(αs.*TPrev, quadSize..., 1)
		# update Transmittance online
	end
	return (cimage, grads)
end

function forward(splatData, cimgview, gt)
	outimage = renderSplats(splatData, cimgview)
	return error(outimage, gt)
end

function loss(img, gt)
	λ = 0.1
	return (1-λ)*sum(abs.(img .- gt))/(2.0*length(img)) + λ*dssim(img, gt)/2.0
end

function errorGrad(cimgview, gtview)
	s = loss(cimgview, gtview)
	@info "loss : " s
	return 2.0.*(((cimgview .- gtview) .> 0) .- 0.5)/length(cimgview)
end



n = 1

staticRot = repeat([1, 0, 0, 1], 1, n);
staticMeans = repeat([0.5, 0.5], 1, n)
staticOpacities = rand(1, n)
staticScales = 0.3.*ones(2, n)
staticColors = rand(3, n)# ([1.0, 0.0, 0.0], 1, n)

using Rotations

function genSplatData(n)
	means = rand(2, n)
	rots = 2.0.*rand(1, n) .- 1.0
	colors = rand(3, n)
	scales = 2.0.*rand(2, n) .- 2.0
	opacities = rand(1, n)
	return GSplatData(means, rots, colors, scales, opacities)
end

"""
function genSplatReference(n)
	means = repeat([0.5, 0.5], 1, n)
	rots = reshape([i*2/((n+1)) for i in 1:n], 1, n) .- 1.0
	colors = channelview(map(x -> RGB(HSV(rand().*repeat([x*1/n], 3)...)), 1:n)) |> collect
	scales = 2.0.*rand(2, n) .- 4.0
	opacities = 0.6.*ones(1, n)
	return GSplatData(means, rots, colors, scales, opacities) 
end
"""

function genSplatReference(n)
	means = repeat([0.5, 0.5], 1, n)
	rots = reshape([i*2/((n+1)) for i in 1:n], 1, n) .- 1.0
	colors = channelview(map(x -> RGB(HSL(repeat([x*1/n], 3)...)), 1:n)) |> collect
	scales = -0.69314718.*[1.0, 2.0].*ones(2, n)
	opacities = ones(1, n)
	return GSplatData(means, rots, colors, scales, opacities)
end

splatDataOriginal = genSplatReference(n)

imgSize = (32, 32)

imgsrc = nothing #"G.jpg"

if imgsrc == nothing
	img = zeros(RGB{N0f8}, imgSize)
	cimgview = permutedims(channelview(img), (2, 3, 1)) .|> float32
	(outimage, grads) = renderSplats(splatDataOriginal, cimgview)
	outimage = colorview(RGB{N0f8},
		n0f8.(clamp.(permutedims(outimage, (3, 1, 2)), 0.0, 1.0))
	)
	save("fontsplat.jpg", outimage)
else
	img = load(imgsrc)
	img = imresize(img, imgSize)
	save("fontsplat.jpg", img)
end

gt = load("fontsplat.jpg")
gtview = permutedims(channelview(gt), (2, 3, 1)) .|> float32

(kernel, cdims) = initKernel(gtview, gtview, windowSize)

using ImageView

gui = imshow_gui((512, 512))
canvas = gui["canvas"]

splatData = genSplatData(n)
lr = 0.1
for i in 0001:20000
	@info "iteration: $(i)"
	target = zeros(RGB{N0f8}, imgSize)
	targetview = permutedims(channelview(target), (2, 3, 1)) .|> float32
	(tmpimageview, grads) = renderSplats(splatData, targetview)
	if i%100 == 0
		tmpimage = colorview(RGB{N0f8}, 
			n0f8.(clamp.(permutedims(tmpimageview, (3, 1, 2)), 0.0f0, 1.0f0))
		)
		imshow!(canvas, tmpimage)
		#save("fontsplat$(i).jpg", tmpimage)
	end
	ΔC = gradient(loss, tmpimageview, gtview)
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
	
	for (idx, gradf) in enumerate(grads)
		(cGrad, oGrad, μGrad, θGrad, sGrad) = gradf(ΔC[1])
		splatData.means[:, idx] .-= (lr.*μGrad)
		splatData.colors[:, idx] .-= lr.*cGrad
		splatData.opacities[:, idx] .-= lr.*oGrad
		splatData.rotations[:, idx] .-= lr.*θGrad
		splatData.scales[:, idx] .-= lr.*diag(sGrad)
		#if any(abs.(splatData.means) .> 1.0)
		#	@warn "means are diverging"
		#end
	end
end

