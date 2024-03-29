
function sigma(x)
	z = exp(x)
	return z/(1 + z)
end


function testColorsGrads()
	n = 2
	αs = rand(1, n)
	cns = rand(3, n);
	ci = zeros(3);

	t = 1.0
	for idx in 1:n
		ci += cns[:, idx].*αs[idx].*t
		t *= (1 - αs[idx])
	end

	cnhats = rand(3, n)

	for itr in 1:2
		local t
		cihat = zeros(3)
		t = 1.0
		for idx in 1:n
			@info "forward tvalue" t
			cihat += cnhats[:, idx].*αs[idx].*t
			t *= (1 - αs[idx])
		end
		Δci = cihat .- ci
		@info "forward tvalue" t
		αsPrev = 0.0
		for idx in n:-1:1
			@info "backward tvalue" t
			α = αs[idx]
			t *= 1/(1.0 - αsPrev)
			Δcn = α.*t.*Δci
			αsPrev = α
			cnhats[:, idx] .-= 0.001.*Δcn
		end
		@info "backward last tvalue" t
		@info "Loss" sum(Δci.^2)
		@info "cn" sum((cnhats .- cns).^2)
	end
	return (cns, cnhats, αs, ci, t)
end

function testRotationGrads()
	θ = pi/2
	θref = pi/3.4

	rot = RotZ(θ)[1:2, 1:2]
	rotRef = RotZ(θref)[1:2, 1:2]


	rGrad = [0 -1; 1 0]*rot*Δc

	lr = 0.01
	for i in 1:1000
		global θ
		Δc = sum((rot - rotRef).^2)
		rGrad = sum([0 -1; 1 0]*rot*Δc)
		θ -= lr*rGrad
		θ %= pi
	end
	
	t = 1.0
	for idx in 1:n
		ci += cns[:, idx].*αs[idx].*t
		t *= (1 - αs[idx])
	end

	cnhats = rand(3, n)

	for itr in 1:2
		local t
		cihat = zeros(3)
		t = 1.0
		for idx in 1:n
			@info "forward tvalue" t
			cihat += cnhats[:, idx].*αs[idx].*t
			t *= (1 - αs[idx])
		end
		Δci = cihat .- ci
		@info "forward tvalue" t
		αsPrev = 0.0
		for idx in n:-1:1
			@info "backward tvalue" t
			α = αs[idx]
			t *= 1/(1.0 - αsPrev)
			Δcn = α.*t.*Δci
			αsPrev = α
			cnhats[:, idx] .-= 0.001.*Δcn
		end
		@info "backward last tvalue" t
		@info "Loss" sum(Δci.^2)
		@info "cn" sum((cnhats .- cns).^2)
	end
	return (cns, cnhats, αs, ci, t)
end



θRef = pi/2.1

θ = rand()

RRef = RotZ(θRef)[1:2, 1:2]
lr = 0.01
for _ in 1:1000
	global θ
	R = RotZ(θ)[1:2, 1:2]
	ΔR = (RRef .- R)[1:2, 1:2]
	ΔGrad = [0 -1; 1 0]*R*ΔR
	Δθ = atan(ΔGrad[2, 1], ΔGrad[2, 2])
	θ -= lr*Δθ
	@info θ
end
