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


function h(x,K)
    x*(1-x/K)
end
function lotka!(du,u,p,t) 
    du[1] = p.r*h(u[1],p.K) - 0.5*u[1]*u[2]
    du[2] =  0.5*u[1]*u[2] - p.m*u[2]
end
println("Generating data")
#data generation parameters
dt=0.1
tspan = (0.0, 14)
tspan_train=(0.0, 14)
u0 = [1.0f0, 1.0f0]
p_= (r= 1.0f0, m=1.0f0, K=10.0f0)
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
normalizer = KolmogorovArnold.softsign # sigmoid(_fast), tanh(_fast), softsign
num_layers=2 #defined just to save into .mat for plotting
layer_width=7
grid_size=5
kan1 =Lux.Chain(
    KDense( 1, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  1, grid_size; use_base_act = true, basis_func, normalizer),
)
# initialize the neural network states and parameters
rng = Random.default_rng()
Random.seed!(rng, 3)

pM, stM = Lux.setup(rng,kan1)
# Define the UDE
function lotka_voltera!(du,u,p,t)
    C, _ = kan1([u[1]],pM,stM)
    du[1] = 1.0*C[1] -  0.5*u[1]*u[2]
    du[2] =  0.5*u[1]*u[2] - 1.0*u[2]
end
#=
function lotka_voltera!(du,u,p,t)
    C, _ = kan1([u[1]],pM,stM)
    du[1] = p.r*C[1] -  0.5*u[1]*u[2]
    du[2] =  0.5*u[1]*u[2] - p.m*u[2]
end
=#
initial_parameters = (NN=pM)#, r=2.0, m=0.5, K=10.0)
model = CustomDerivatives(data, lotka_voltera!, initial_parameters)
##Plotting
#Bounds on the interaction plot
x = range(0, 3, length=20)
# True interaction function   
true_h = [h(i,p_.K) for i in x]
#How fast the interaction plot spis
spinning_rate=0.2
sol_max_x = maximum(data.prey)
static_data = StaticData_1D(x,true_h,tspan_train, u0, spinning_rate, sol_max_x)


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
    print(pM)
    #println("Time to render graphics")
    #UniversalDiffEq.plot_state_estimates(model)
    @time display(save_training_frame_1d(static_data, model, kan1, pM, i*iters_per_loop,l,l_test, training_dir; save=SAVE_ON))
end