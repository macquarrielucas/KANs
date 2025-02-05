# PACKAGES AND INCLUSIONS
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, LinearAlgebra
using Flux: mae, update!, mean
using Random
using ModelingToolkit
using MAT
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using ForwardDiff
using Optimisers
using Zygote, ProgressBars, Printf
pythonplot()
#my plotting file
include("plotting_functions.jl")
# DIRECTORY
dir         = @__DIR__
dir         = dir*"/"
cd(dir)
add_path    = "test/"

SAVE_ON = false
if SAVE_ON
    # Create directories if needed
    folder_count = 0
    training_dir = joinpath(dir, add_path, "training_frames_$folder_count")
    
    # While the directory exists and is not empty, try the next folder.
    while isdir(training_dir) && !isempty(readdir(training_dir))
        global folder_count += 1
        global training_dir = joinpath(dir, add_path, "training_frames_$folder_count")
    end
    
    # If the directory doesn't exist, create it. If it exists but is empty, you can use it.
    if !isdir(training_dir)
        println("Making directory ", training_dir)
        mkdir(training_dir)
    else
        println("Using existing empty directory ", training_dir)
    end
else
    training_dir = nothing
end
#=
mkpath(join(dir,add_path,"figs"))
mkpath(join(dir,add_path,"checkpoints"))
mkpath(join(dir,add_path,"training_frames"))
=#
# KAN PACKAGE LOAD
include("../Lotka-Volterra/src/KolmogorovArnold.jl")
using .KolmogorovArnold

#Random
rng = Random.default_rng()
#Random.seed!(rng, 0)

#   Here we will use KAN-ODEs to try to learn a hidden term in the lotka voltera
#   model :$ x' = x - h(x,y) \ y' = h(x,y) - y :$ where h(x,y) = 0.5xy.
# 
#   Our workflow will be as follows
# 
#     •  Generate data for the model by simulating the true ode
# 
#     •  define the KAN we will use for simulation
# 
#     •  Insert the known terms and the KAN into the ODE Problem
# 
#     •  define the loss by integrating the ODE problem and comparing it
#        with the true data
# 
#     •  train the model

#   We'll first deal with the data generation. We'll simply use ODEProblem and
#   solve from DifferentialEquations, then graph our solution to make sure
#   everything is as it should be.



function h(x,y) 
    0.5*x*y
end
function lotka!(du,u,p,t) 
    du[1] = u[1] - h(u[1],u[2])
    du[2] =  h(u[1],u[2]) - u[2]
end

#data generation parameters
dt=0.1
tspan = (0.0, 26)
tspan_train=(0.0, 14)
u0 = [1.0f0, 1.0f0]
p_=[]

saveat_points = 0.0:dt:tspan[2]
prob = ODEProblem(lotka!, u0,tspan,p_)

#generate training data, split into train/test
solution = solve(prob, Tsit5(), saveat = saveat_points, verbose = false)
end_index=Int64(floor(length(solution.t)*tspan_train[2]/tspan[2])) + 1
t = solution.t #full dataset
t_train=t[1:end_index] #training cut
X = Array(solution)
Xn = deepcopy(X) 
display(plot(solution))


#   Next we'll define the KAN.

basis_func = rbf      # rbf, rswaf
normalizer = tanh_fast # sigmoid(_fast), tanh(_fast), softsign

# Define KAN-ODEs
###layer_width and grid_size can be modified here to replicate the testing in section A2 of the manuscript

num_layers=3 #defined just to save into .mat for plotting
layer_width=10
grid_size=7
kan1 = Lux.Chain(
    KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  1, grid_size; use_base_act = true, basis_func, normalizer),
) 
pM, stM  = Lux.setup(rng, kan1)
pM_data     = getdata(ComponentArray(pM))
pM_axis     = getaxes(ComponentArray(pM))
p = ComponentArray(pM_data, pM_axis) 
#p = (deepcopy(pM_data))./1e5 ;


#   and construct the KAN-ODE. We'll also define the prediction function which
#   will simulate the KAN-ODEs

# CONSTRUCT KAN-ODES
function kanode!(du, u, p, t)
    kan1_(x) = kan1(x, ComponentArray(p,pM_axis), stM)[1][1]
    du[1] = u[1] - kan1_(u)
    du[2] = kan1_(u) - u[2]
end

# PREDICTION FUNCTION
function predict(p)
    prob = ODEProblem(kanode!, u0, tspan_train,p)
    sol = solve(prob, Tsit5(), verbose = false, saveat = t_train);
end
#Prediction function over the test set.
function predict_test(p)
    prob = ODEProblem(kanode!, u0, tspan,p)
    sol = solve(prob, Tsit5(), verbose = false, saveat = t);
end

#   To define the loss function we'll use the MSE loss :$\mathcal{L}1(\theta) =
#   MSE(u^{\text{KAN}}(t, \theta), u^{\text{obs}}(t)) = \frac{1}{N}\sum{i=1}^N
#   \lVert u^{\text{KAN}}(ti, \theta) - u^{\text{obs}}(ti) \rVert^2 :$ We'll
#   define two functions: one for the training loss and one for the test loss,
#   the latter being used for model validation later on.

function loss_train(p)
    mean(abs2, Xn[:, 1:end_index] .- Array(predict(p)))
end

function loss_test(p)
    mean(abs2, Xn .- Array(predict_test(p)))
end

#   Now that we've defined our model, loss, and make prediction functions, we
#   may make the training loop.




##HYPERPARAMETERS
opt = Flux.Adam(5e-3)
N_iter = 1000

##DATA FOR PLOTTING
# Create grid for interaction function visualization
x = range(0, 3, length=20)
y = range(0, 3, length=20)
xy = [(i,j) for i in x, j in y]
# True interaction function   
true_h = [h(i,j) for (i,j) in xy]
static_data = StaticData(x,y,xy,true_h,solution,tspan_train, u0)

l = []
l_test=[]
p_list = []
iterator = ProgressBar(1:N_iter)
@time for i in iterator
    # GRADIENT COMPUTATION
    grad = Zygote.gradient(loss_train, p)[1]

    # UPDATE WITH ADAM OPTIMIZER
    update!(opt, p, grad)


    # CALLBACK
    loss_curr=deepcopy(loss_train(p))
    loss_curr_test=deepcopy(loss_test(p))
    set_description(iterator, string("Iter:" ,i, "| Loss:", @sprintf("%.2f", loss_curr), "|",
                            "Test_Loss:", @sprintf("%.2f", loss_curr_test), "|"))
    append!(l, [loss_curr])
    append!(l_test, [loss_curr_test])
    #append!(p_list, [deepcopy(p)])
    if i%10 == 0
        @time display(save_training_frame_2d(static_data,p,i,l, l_test, training_dir; save=SAVE_ON))
    end
    # SAVE
    #callback(i)
end