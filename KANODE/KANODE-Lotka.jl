using Random, Lux, LinearAlgebra
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using BenchmarkTools
using OrdinaryDiffEq, Plots, DiffEqFlux, ForwardDiff
using Flux: Adam, mae, update!
using Flux
using Optimisers
using MAT
using Plots
using ProgressBars
using Zygote: gradient as Zgrad

# Load the KAN package from https://github.com/vpuri3/KolmogorovArnold.jl
include("src/KolmogorovArnold.jl")
using .KolmogorovArnold
#load the activation function getter (written for this project, see the corresponding script):
include("Activation_getter.jl")

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

#data generation parameters
timestep=0.1
n_plot_save=1000
rng = Random.default_rng()
Random.seed!(rng, 0)
tspan = (0.0, 14)
tspan_train=(0.0, 3.5)
u0 = [1, 1]
p_ = Float32[1.5, 1, 1, 3]
prob = ODEProblem(lotka!, u0, tspan, p_)

#generate training data, split into train/test
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = timestep)
end_index=Int64(floor(length(solution.t)*tspan_train[2]/tspan[2]))
t = solution.t #full dataset
t_train=t[1:end_index] #training cut
#NOTE: What are these?
X = Array(solution)
Xn = deepcopy(X) 

basis_func = rbf      # rbf, rswaf
normalizer = tanh_fast # sigmoid(_fast), tanh(_fast), softsign 
##Not sure what this is? It seems like this normalizes the inputs 
##to be between -1,1,/0,1 but i dont quite see for sure where.


###layer_width and grid_size can be modified here to replicate the testing in section A2 of the manuscript

num_layers=2 #defined just to save into .mat for plotting
layer_width=10
grid_size=5
kan1 = Lux.Chain(
    KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  2, grid_size; use_base_act = true, basis_func, normalizer),
)
pM , stM  = Lux.setup(rng, kan1)

l = []
l_test=[]
p_list = []
pM_axis = getaxes(ComponentArray(pM))
pM_data = getdata(ComponentArray(pM))
p = (deepcopy(pM_data))./1e5

train_node = NeuralODE(kan1, tspan_train, Tsit5(), saveat = t_train);
#TODO: Understand why this is necessary
train_node_test = NeuralODE(kan1, tspan, Tsit5(), saveat = t); #only difference is the time span
function predict(p)
    Array(train_node(u0, p, stM)[1])
end

#regularization loss (see Eq. 12 in manuscript )
#act_reg must be γ_sp in the paper.

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
function loss(p)
    loss_temp=mean(abs2, Xn[:, 1:end_index].- predict(ComponentArray(p,pM_axis)))
    if sparse_on==1
        loss_temp+=reg_loss(p, 5e-4, 0) #if we have sparsity enabled, add the reg loss
    end
    return loss_temp
end

function predict_test(p)
    Array(train_node_test(u0, p, stM)[1])
end

function loss_train(p)
    mean(abs2, Xn[:, 1:end_index].- predict(ComponentArray(p,pM_axis)))
end
function loss_test(p)
    mean(abs2, Xn .- predict_test(ComponentArray(p,pM_axis)))
end


# TRAINING
du = [0.0; 0.0]
optimizer = Flux.Adam(5e-4)
sparse_on = 0 #<-- this will determine if the lose function will include a sparsity term
N_iter = 10
i_current = 1

print("Starting loop")
##Actual training loop:
#tqdm is part of the progress bar package
iters=tqdm(1:N_iter-i_current)
 for i in iters
    global i_current
    
    # gradient computation
    grad = Zgrad(loss, p)[1]

    #model update
    update!(optimizer, p, grad)

    #loss metrics
    #Use deepcopy 
    loss_curr=deepcopy(loss_train(p))
    loss_curr_test=deepcopy(loss_test(p))
    append!(l, [loss_curr])
    append!(l_test, [loss_curr_test])
    append!(p_list, [deepcopy(p)])

    set_description(iters, string("Loss:", loss_curr))
    i_current = i_current + 1

    #=
    if i%n_plot_save==0
        plot_save(l, l_test, p_list, i)
    end
    =#
    
end

