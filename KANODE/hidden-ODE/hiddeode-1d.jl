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

include("plotting_functions.jl")
# DIRECTORY
dir         = @__DIR__
dir         = dir*"/"
cd(dir)
add_path    = "test/"
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
#Random.seed!(rng, 1)

function lotka!(du,u,p,t) 
    du[1] = u[1]*(1-u[1]) - 0.5*u[1]*u[2]
    du[2] = 0.5*u[1]*u[2] - 0.03*u[2]
end

#data generation parameters
dt=0.1
tspan_test = (0.0, 500)
tspan_train=(0.0, 100)
u0 = [1.0f0, 1.0f0]
p_=[]
prob = ODEProblem(lotka!, u0,tspan_test,p_)

#generate training data, split into train/test
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = dt, verbose = false)
end_index=Int64(floor(length(solution.t)*tspan_train[2]/tspan_test[2])) + 1
t_test = solution.t #full dataset
t_train=t_test[1:end_index] #training cut
X = Array(solution)
Xn = deepcopy(X) 
plot(solution)


#   Next we'll define the KAN.

basis_func = rbf      # rbf, rswaf
normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign

# Define KAN-ODEs
###layer_width and grid_size can be modified here to replicate the testing in section A2 of the manuscript


layer_width=5
grid_size=5
#This KAN looks like
# phi1(x) + phi2(x) + phi3(x) + ... + phi10(x) 
kan1 = Lux.Chain(
    KDense( 1, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  1, grid_size; use_base_act = true, basis_func, normalizer),
)
pM , stM  = Lux.setup(rng, kan1)
pM_data     = getdata(ComponentArray(pM))
pM_axis     = getaxes(ComponentArray(pM))
p = ComponentArray(pM_data, pM_axis) 
#p = (deepcopy(pM_data))./1e5 ;


#   and construct the KAN-ODE. We'll also define the prediction function which
#   will simulate the KAN-ODEs

# CONSTRUCT KAN-ODES
function kanode!(du, u, p, t)
    kan1_(x) = kan1([x], p, stM)[1][1]
    du[1] = kan1_(u[1]) - 0.5*u[1]*u[2]
    du[2] = 0.5*u[1]*u[2] - 0.03*u[2]
end

# PREDICTION FUNCTION
function predict(p)
    prob = ODEProblem(kanode!, u0, tspan_train,p, saveat=dt)
    sol = solve(prob, Tsit5(), verbose = false);
end
#Prediction function over the test set.
function predict_test(p)
    prob = ODEProblem(kanode!, u0, tspan_test,p, saveat=dt)
    sol = solve(prob, Tsit5(), verbose = false);
end

# LOSS FUNCTION


function reg_loss(p, act_reg=1.0, entropy_reg=1.0)
    l1_temp=(abs.(p))
    activation_loss=sum(l1_temp)
    #This entropy was not mentioned in the paper i believe,
    #so i assuming its some optional thing they played with.
    entropy_temp=l1_temp/activation_loss
    entropy_loss=-sum(entropy_temp.*log.(entropy_temp))
    total_reg_loss=activation_loss*act_reg+entropy_loss*entropy_reg
    return total_reg_loss
end

#overall loss
sparse_on = 0
function loss_train(p)
    loss_temp=mean(abs2, Xn[:, 1:end_index].- predict(p))
    if sparse_on==1
        loss_temp+=reg_loss(p, 5e-4, 0) #if we have sparsity enabled, add the reg loss
    end
    return loss_temp
end
#=
function loss_train(p)
    mean(abs2, Xn[:, 1:end_index] .- Array(predict(p)))
end
=#
function loss_test(p)
    mean(abs2, Xn .- Array(predict_test(p)))
end

# Create grid for interaction function visualization

sol_max = maximum(vcat(solution.u...)) #Max of x for plotting
sol_max_x =maximum([x[1] for x in solution.u]) 
x = collect(range(-1, sol_max_x+1, length=40))
true_sol = solution  # Higher resolution for smooth plot
#y = range(0, 3, length=20)
#xy = [(i,j) for i in x, j in y]

# True interaction function
#true_h = [0.5*i*j for (i,j) in xy]

# KAN prediction
#kan_h = [kan1([i,j], ComponentArray(p,pM_axis), stM)[1][] for (i,j) in xy]



SAVE_ON = false
training_dir = nothing
if SAVE_ON
    # Create directories if needed
    folder_count = 0
    training_dir = joinpath(dir, add_path, "training_frames_$folder_count")
    is_empty = isempty(readdir(training_dir))
    #If its already a directory and its not empty, try the next folder
    while isdir(training_dir) && !is_empty 
        global folder_count += 1 
        global training_dir = joinpath(dir, add_path, "training_frames_$folder_count")
    end
    if !is_empty
        print("Making directory ", string(training_dir))
        mkdir(training_dir)
    end
end
#opt = Flux.Momentum(1e-3, 0.9)
opt = Flux.Adam(1e-4)
N_iter = 10000
iterator = ProgressBar(1:N_iter)
l = []
l_test=[]
p_list = []

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

    if i%10 == 0 || i==1
        print(training_dir)
        @time display(save_training_frame_1d(p,i, l,l_test,training_dir; save=SAVE_ON))
    end
    # SAVE
    #callback(i)
end