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
pythonplot()

# DIRECTORY
dir         = @__DIR__
dir         = dir*"/"
cd(dir)
fname       = "Fisher_KPP_Source"
add_path    = "test/"
mkpath(dir*add_path*"figs")
mkpath(dir*add_path*"checkpoints")
mkpath(dir*add_path*"results")
mkpath(dir*add_path*"training_frames")
# KAN PACKAGE LOAD
include("../Lotka-Volterra/src/KolmogorovArnold.jl")
using .KolmogorovArnold

#Random
rng = Random.default_rng()
Random.seed!(rng, 0)

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

function lotka!(du,u,p,t) 
    du[1] = u[1] - 0.5*u[1]*u[2]
    du[2] = 0.5*u[1]*u[2] - u[2]
end

#data generation parameters
dt=0.1
tspan = (0.0, 14)
tspan_train=(0.0, 3.5)
u0 = [1, 1]
p_=[]
prob = ODEProblem(lotka!, u0,tspan,p_)

#generate training data, split into train/test
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = dt, verbose = false)
end_index=Int64(floor(length(solution.t)*tspan_train[2]/tspan[2])) + 1
t = solution.t #full dataset
t_train=t[1:end_index] #training cut
X = Array(solution)
Xn = deepcopy(X) 
plot(solution)


#   Next we'll define the KAN.

basis_func = rbf      # rbf, rswaf
normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign

# Define KAN-ODEs
###layer_width and grid_size can be modified here to replicate the testing in section A2 of the manuscript

num_layers=2 #defined just to save into .mat for plotting
layer_width=10
grid_size=5
kan1 = Lux.Chain(
    KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
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
    kan1_(x) = kan1(x, ComponentArray(p,pM_axis), stM)[1][1]
    du[1] = u[1] - kan1_(u)
    du[2] = kan1_(u) - u[2]
end
#TODO: LUCAS NOTE FOR TOMORROW JAN 29. run the code then see the error. 
# something is going wrong with dimension mismatch off by one? Note that
# the size of the two arrays in the error output are 35, which is the size of
#t_train....

# PREDICTION FUNCTION
function predict(p)
    prob = ODEProblem(kanode!, u0, tspan_train,p, saveat=dt)
    sol = solve(prob, Tsit5(), verbose = false);
end
#Prediction function over the test set.
function predict_test(p)
    prob = ODEProblem(kanode!, u0, tspan,p, saveat=dt)
    sol = solve(prob, Tsit5(), verbose = false);
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

opt = Flux.Adam(5e-4)
using Zygote, ProgressBars, Printf
N_iter = 1000
iterator = ProgressBar(1:N_iter)
function save_training_frame(p, iter, training_dir; save=false)

    # Create grid for interaction function visualization
    x = range(0, 3, length=20)
    y = range(0, 3, length=20)
    xy = [(i,j) for i in x, j in y]
    
    # True interaction function
    true_h = [0.5*i*j for (i,j) in xy]
    
    # KAN prediction
    kan_h = [kan1([i,j], ComponentArray(p,pM_axis), stM)[1][] for (i,j) in xy]

    # Get predictions for trajectories
    true_sol = solution  # Higher resolution for smooth plot
    kan_sol = predict_test(p)
    kan_train_sol = predict(p)
    
    # Create 2x2 grid plot
    plt = plot(layout=(2,2), size=(1600, 1200), titlefontsize=12)
    
    # Top row: Interaction function surfaces
    plot!(plt[1], x, y, true_h, st=:surface, 
          title="True Interaction: h(x,y) = 0.5xy",
          xlabel="x", ylabel="y", zlabel="h(x,y)",
          zlim=(0,4.5), camera=(30, 30))
    
    plot!(plt[2], x, y, kan_h, st=:surface, 
          title="KAN Learned Interaction (Iteration $iter)",
          xlabel="x", ylabel="y", zlabel="h(x,y)",
          zlim=(0,4.5), camera=(30, 30))
    
    # Bottom left: Phase plane
    plot!(plt[3], true_sol[1,:], true_sol[2,:], 
          label="True Solution", lw=2, color=:black,
          xlabel="x", ylabel="y",
          xlim=(0,4.5),ylim=(0,4.5),
           title="Phase Plane")
    plot!(plt[3], kan_sol[1,:], kan_sol[2,:], 
          label="KAN Prediction", lw=2, color=:red, linestyle=:dash)
    scatter!(plt[3], [u0[1]], [u0[2]], color=:blue, label="Initial Condition")
    
    # Bottom right: Time series comparison
    plot!(plt[4], true_sol.t, true_sol[1,:], label="True x", color=:blue, lw=2)
    plot!(plt[4], true_sol.t, true_sol[2,:], label="True y", color=:green, lw=2)
    plot!(plt[4], kan_sol.t, kan_sol[1,:], label="KAN x", color=:blue, linestyle=:dash, lw=2)
    plot!(plt[4], kan_sol.t, kan_sol[2,:], label="KAN y", color=:green, linestyle=:dash, lw=2)
    vspan!(plt[4], [tspan_train[1], tspan_train[2]], alpha=0.2, color=:gray, label="Training Region")
    plot!(plt[4], 
        xlabel="Time", ylabel="Population",
        ylim=(0,6),
        title="Time Series Comparison", legend=:topright)
    if save
        # Save figure with iteration number
        savefig(plt, joinpath(training_dir, "frame_$(lpad(iter, 5, '0')).png"))
    end
    return plt
end

# Create directories if needed
training_dir = dir*add_path*"training_frames"*"_" * string(round(Int, time()))
mkpath(training_dir)

for i in iterator
    
    # GRADIENT COMPUTATION
    grad = Zygote.gradient(loss_train, p)[1]

    # UPDATE WITH ADAM OPTIMIZER
    update!(opt, p, grad)


    # CALLBACK
    loss_curr=deepcopy(loss_train(p))
    loss_curr_test=deepcopy(loss_test(p))
    set_description(iterator, string("Iter:" ,i, "| Loss:", @sprintf("%.2f", loss_curr), "|",
                            "Test_Loss:", @sprintf("%.2f", loss_curr_test), "|"))

    if i%10 == 0
        display(save_training_frame(p,i, training_dir; save=true))
    end
    # SAVE
    #callback(i)
end