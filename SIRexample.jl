using DifferentialEquations
using Random
using Plots

# Define the SIR model
function sir_model!(du, u, p, t)
    S, I, R = u
    β, γ = p
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

# Parameters
β = 0.3  # Infection rate
γ = 0.1  # Recovery rate
S0 = 0.99  # Initial susceptible fraction
I0 = 0.01  # Initial infected fraction
R0 = 0.0   # Initial recovered fraction
u0 = [S0, I0, R0]
p = [β, γ]
tspan = (0.0, 100.0)

# Solve the differential equation
tspan = (0.0, 100.0)  # Keep the original timespan
t = range(tspan[1], tspan[2], length=50)  # Create a timespan with 200 points
prob = ODEProblem(sir_model!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=t)  # Solve with specified time points
# Add noise to simulate time series data
Random.seed!(42)  # For reproducibility
time_series = sol.t
S_noisy = sol[1, :] .+ 0.01 .* randn(length(sol.t))
I_noisy = sol[2, :] .+ 0.01 .* randn(length(sol.t))
R_noisy = sol[3, :] .+ 0.01 .* randn(length(sol.t))

# Plot the noisy time series data
scatter(time_series, S_noisy, label="Susceptible (S)", lw=2)
scatter!(time_series, I_noisy, label="Infected (I) ", lw=2)
scatter!(time_series, R_noisy, label="Recovered (R)", lw=2)
plot!(legend=:outerbottom, legendcolumns=3)
xlabel!("Time")
ylabel!("Population Fraction")
title!("SIR Model with Noise")