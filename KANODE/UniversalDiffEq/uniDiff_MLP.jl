using Lux
using Random
using Zygote 
using DataFrames
using OrdinaryDiffEq
using UniversalDiffEq
using ProgressBars
function lotka!(du,u,p,t) 
    du[1] = p.r*u[1] - 0.5f0*u[1]*u[2]
    du[2] = 0.5f0*u[1]*u[2] - p.m*u[2]
end

#data generation parameters
dt=0.1
tspan = (0.0, 14)
tspan_train=(0.0, 3.5)
u0 = [1.0f0, 1.0f0]
p_= ( r= 1.0f0, m=1.0f0)
prob = ODEProblem(lotka!, u0,tspan,p_)
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = dt, verbose = false )

# Extract solution data
times = solution.t  # Time points
u1 = solution[1, :]  # First variable (e.g., prey population)
u2 = solution[2, :]  # Second variable (e.g., predator population)

# Create a DataFrame
data = DataFrame(time=times, prey=u1, predator=u2)

# Build the neural network with Lux.Chain
dims_in = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# initialize the neural network states and parameters

rng = Random.default_rng()
pM, stM = Lux.setup(rng,NN)

function lotka_voltera!(du,u,p,t)
    C, _ = NN(u,p.NN,stM)
    du[1] = 0 #p.r*u[1] - C[1]
    du[2] = 0 #C[1] - p.m*u[2]
end
  
initial_parameters = (NN=pM, r=1.0, m=0.5)
model = CustomDerivatives(data, lotka_voltera!, initial_parameters)

UniversalDiffEq.gradient_descent!(model,verbose = true)
#UniversalDiffEq.phase_plane(model)
UniversalDiffEq.plot_state_estimates(model)