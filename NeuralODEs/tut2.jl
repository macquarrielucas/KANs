using ComponentArrays, DiffEqFlux, Optimization, OptimizationOptimisers, OrdinaryDiffEq,
      LinearAlgebra, Random
k, α, β, γ = 1, 0.1, 0.2, 0.3
tspan = (0.0, 10.0)

function dxdt_train(du, u, p, t)
    du[1] = u[2]
    du[2] = -k * u[1] - α * u[1]^3 - β * u[2] - γ * u[2]^3
end

u0 = [1.0, 0.0]
ts = collect(0.0:0.1:tspan[2])
prob_train = ODEProblem{true}(dxdt_train, u0, tspan)
data_train = Array(solve(prob_train, Tsit5(); saveat = ts))