#This code is used to debug some of the UDE packages wierd behaviour
using Lux, LuxCore
using Flux: mae, update!, mean
using Random
using Zygote 
using OrdinaryDiffEq
using UniversalDiffEq, DataFrames
using ProgressBars, Printf, Plots
using Optimization, Optimisers


##Ground Truth Data Generation ########
function lotka!(du,u,p,t) 
    du[1] = p.r*u[1] - 0.5f0*u[1]*u[2]
    du[2] = 0.5f0*u[1]*u[2] - p.m*u[2]
end

#data generation parameters
dt=0.5
tspan = (0.0, 14)
tspan_train=(0.0, 14)
u0 = [1.0f0, 1.0f0]
p_= (r= 1.0f0, m=1.0f0)
prob = ODEProblem(lotka!, u0, tspan, p_)

solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = dt, verbose = false )
# Extract solution data
times = solution.t  # Time points
u1 = solution[1, :]  # First variable (e.g., prey population)
u2 = solution[2, :]  # Second variable (e.g., predator population)
# Create a DataFrame
data = DataFrame(time=times, prey=u1, predator=u2)

########Neural Network Model definition#####
dims_in = 2; dims_out = 1; hidden = 1
Neural_Network = Lux.Chain(Lux.Dense(dims_in,hidden), Lux.Dense(hidden,dims_out))
# initialize the neural network states and parameters
rng = Random.default_rng()
pM, stM = Lux.setup(rng,Neural_Network)
# Define the UDE
function lotka_voltera!(du,u,p,t)
    C, _ = Neural_Network(u,pM,stM)
    du[1] = p.r*u[1] - C[1]
    du[2] = C[1] - p.m*u[2]
end

initial_parameters = (NN=pM, r=2.0, m=0.5)
function dudt!(du,u,p,t)
    du[1]=0
    du[2]=0
end
model = CustomDerivatives(data, dudt!, initial_parameters)
##Training
N_iter = 10000
iterator = ProgressBar(1:N_iter)
@time for i in 1:N_iter#iterator

    #Update model 
    UniversalDiffEq.train!(model, 
                            loss_function = "multiple shooting",
                            verbose = false,
                            regularization_weight = 0,
                            optimizer = "ADAM",
                            optim_options = (maxiter = 2, step_size = 1e-2)
                            )

    display(UniversalDiffEq.plot_state_estimates(model))
end