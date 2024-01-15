
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


