using Lux, LuxCore
using Random
using Zygote 
using DataFrames
using OrdinaryDiffEq
using UniversalDiffEq
using ProgressBars
using Flux: mae, update!, mean
using Optimization
using Optimisers
include("../Lotka-Volterra/src/KolmogorovArnold.jl")
using .KolmogorovArnold
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

# Define KAN-ODEs
###layer_width and grid_size can be modified here to replicate the testing in section A2 of the manuscript

num_layers=2 #defined just to save into .mat for plotting
layer_width=10
grid_size=5
kan1 = Lux.Chain(
    KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  1, grid_size; use_base_act = true, basis_func, normalizer),
)
# initialize the neural network states and parameters

rng = Random.default_rng()
pM, stM = Lux.setup(rng,kan1)

function lotka_voltera!(du,u,p,t)
    C, _ = kan1(u,p.NN,stM)
    du[1] = p.r*u[1] - C[1]
    du[2] = C[1] - p.m*u[2]
end

initial_parameters = (NN=pM, r=1.0, m=0.5)
model = CustomDerivatives(data, lotka_voltera!, initial_parameters)

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

N_iter = 100
iterator = ProgressBar(1:N_iter)
iters_per_loop = 10
@time for i in iterator

    #Update model
    # set optimization problem 
    target = (x,p) -> model.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, model.parameters)

    # run optimizer for 10 steps
    sol = Optimization.solve(optprob, 
                                Optimisers.Adam(0.05), 
                                maxiters = iters_per_loop)
    
    # assign parameters to model 
    model.parameters = sol.u


    # CALLBACK
    loss_curr=deepcopy(model.loss_function(model.parameters))
    #loss_curr_test=deepcopy(loss_test(p))
    set_description(iterator, string("Iter:", i, "| Loss:", @sprintf("%.2f", loss_curr), "|"))

    #display(save_training_frame(p, i, training_dir; save=false))
    plt = UniversalDiffEq.plot_state_estimates(model)
    display(plt)
    savefig(plt, "frames/frame_$(lpad(i, 5, '0'))")
    # SAVE
    #callback(i)
end
