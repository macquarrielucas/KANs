using Lux, LuxCore
using Flux: mae, update!, mean
using Random
using Zygote 
using OrdinaryDiffEq
using UniversalDiffEq, DataFrames
using ProgressBars, Printf, Plots
using Optimization, Optimisers

include("../Lotka-Volterra/src/KolmogorovArnold.jl")
using .KolmogorovArnold
include("plotting_functions.jl")


function h(x,y)
    0.5*x*y
end
function lotka!(du,u,p,t) 
    du[1] = p.r*u[1] - h(u[1],u[2])
    du[2] = h(u[1],u[2]) - p.m*u[2]
end
println("Generating data")
#data generation parameters
dt=0.1
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

println("Setting up KAN")
##Model definition
basis_func = KolmogorovArnold.rbf      # rbf, rswaf
normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign
num_layers=2 #defined just to save into .mat for plotting
layer_width=7
grid_size=5
kan1 =Lux.Chain(
    KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  1, grid_size; use_base_act = true, basis_func, normalizer),
)
# initialize the neural network states and parameters
rng = Random.default_rng()
Random.seed!(rng, 3)

pM, stM = Lux.setup(rng,kan1)
# Define the UDE
function lotka_voltera!(du,u,p,t)
    C, _ = kan1(u,pM,stM)
    du[1] = p.r*u[1] - C[1]
    du[2] = C[1] - p.m*u[2]
end

initial_parameters = (NN=pM, r=2.0, m=0.5)
model = CustomDerivatives(data, lotka_voltera!, initial_parameters)
##Plotting
#Bounds on the interaction plot
x = range(0, 3, length=20)
y = range(0, 3, length=20)
xy = [(i,j) for i in x, j in y]
# True interaction function  
true_h = [h(i,j) for (i,j) in xy]
#How fast the interaction plot spis
spinning_rate=0.0
static_data = StaticData_2D(x,y,xy,true_h,tspan_train, u0, spinning_rate)


##Training
N_iter = 100
iterator = ProgressBar(1:N_iter)
iters_per_loop = 5
l = []
l_test=[]
p_list = []
SAVE_ON = false
#Where to store the frames if saving
if SAVE_ON
    training_dir = find_frame_directory()
else
    training_dir=""
end
print("Starting Training")
@time for i in iterator

    #Update model
    #=
    # set optimization problem 
    target  = (x,p) -> model.loss_function(x)
    adtype  = Optimization.AutoZygote()
    optf    = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, model.parameters)

    # run optimizer for 10 steps
    print("Time to train $iters_per_loop steps")
    @time sol = Optimization.solve(optprob, 
                                Optimisers.Adam(0.05), 
                                maxiters = iters_per_loop+1)
    
    # assign parameters to model 
    model.parameters = sol.u
    =#

    UniversalDiffEq.train!(model, 
                            loss_function = "multiple shooting",
                            verbose = false,
                            regularization_weight = 0,
                            optimizer = "ADAM",
                            optim_options = (maxiter = 2, step_size = 1e-2)
                            )


    loss_curr=deepcopy(model.loss_function(model.parameters))
    loss_curr_test=0

    append!(l, [loss_curr])
    append!(l_test, [loss_curr_test])

    # CALLBACK
    #print("Time to print loss")
    set_description(iterator, string("Iter:", i, "| Loss:", @sprintf("%.2f", loss_curr), "|"))
    #println("Time to render graphics")
    #UniversalDiffEq.plot_state_estimates(model)
    @time display(save_training_frame_2d(static_data, model, kan1, pM, i*iters_per_loop,l,l_test, training_dir; save=SAVE_ON))
end