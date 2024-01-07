using Flux

using Images
using ImageView
using LinearAlgebra

struct GSplatData
	means
	rotations
	scales
end

n = 10

using Rotations

function genSplatData()
	means = rand(2, n)
	rots = rand(4, n)
	scales = rand(2, n)
	return GSplatData(means, rots, scales) 
end

splatData = genSplatData()
splats = splatData
idx = 1


gt = load(joinpath(
	ENV["HOME"],
	"Desktop",
	"six.png"
)) |> channelview

image = zeros(RGBA{N0f8}, size(gt)[2:3])
cimgview = channelview(image)
cimgview[4, :, :] .= 1.0


function drawSplat!(cov2d, quadview, quadGrads)
	sz = size(quadview)[2:end]
	idxs = CartesianIndices(Tuple([1:sz[1], 1:sz[2]])) |> collect
	intensities = ones(N0f8, sz...)
	invCov2d = inv(cov2d)
	for idx in idxs
		delta = [idx.I[1] - sz[1]/2, idx.I[2] - sz[2]]./[sz...]
		dist = (delta |> adjoint)*invCov2d*delta
		intensity = quadview[4, idx] .* (exp(-dist))
		intensities[idx] = intensity |> N0f8
	end
	quadview[1, :, :] .*= intensities
	quadview[2, :, :] .*= intensities
	quadview[3, :, :] .*= intensities
	quadview[4, :, :] .*= 1 .- intensities
	gradGrads = quadview[4, :, :].*intensities
end

function renderSplats!(splats, cimage)
	nPoints = splats.means |> size |> last
	for idx in 1:nPoints
		point = [size(cimage)[2:end]...].*splats.means[:, idx]
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
		quadview = view(cimage, :, UnitRange(bb[1, :]...), UnitRange(bb[2, :]...))
		#quadview = bbview |> channelview
		quadGrads = similar(quadview)
		drawSplat!(cov2d, quadview, grads)
	end 
	#cimage[:, :, :] .= (1.0 |> N0f8) .- cimage[:, :, :]
end

renderSplats!(splatData, cimgview)

function error(img, gt)
	i = channelview(img)
	g = channelview(gt)
	return sum(((1.0 .- i) .- g).^2)
end


using Flux

grads = gradient(error, cimgview, gt)

function gradColor()
	
end

imshow(image)

