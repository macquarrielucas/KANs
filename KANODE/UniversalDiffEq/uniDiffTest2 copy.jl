#This code is used to debug some of the UDE packages wierd behaviour
using Lux, LuxCore
using Flux: mae, update!, mean
using Random
using Zygote 
using OrdinaryDiffEq
using UniversalDiffEq, DataFrames
using ProgressBars, Printf, Plots
using Optimization, Optimisers


## Plotting#########
#This will plot a 2x2 grid graphing 
#1. the true connecting function h(x,y) = 0.5xy, 
#2. the Neural Network, 
#3. phase plane including the ground truth solution and the estimatedsolution, 
#4. and time series of the ground truth solution and the estimated solution.


# Create grid for interaction function visualization
x = range(0, 3, length=20)
y = range(0, 3, length=20)
xy = [(i,j) for i in x, j in y]
# True interaction function
true_h = [0.5*i*j for (i,j) in xy]

function save_training_frame(model, p, iter, training_dir; save=false)
    # Create 2x2 grid plot
    plt = plot(layout=(2,2), size=(1600, 1200), titlefontsize=12)

    #prediction
    nn_h = [Neural_Network([i,j], pM, stM)[1][] for (i,j) in xy]

    # Get predictions for trajectories
    nn_sol = model.parameters.uhat
    

    # Top row: Interaction function surfaces
    plot!(plt[1], x, y, true_h, st=:surface, 
          title="True Interaction: h(x,y) = 0.5xy",
          xlabel="x", ylabel="y", zlabel="h(x,y)",
          zlim=(0,4.5), camera=(iter, 30))
    
    plot!(plt[2], x, y, nn_h, st=:surface, 
          title="Learned Interaction (Iteration $iter)",
          xlabel="x", ylabel="y", zlabel="h(x,y)",
          zlim=(0,4.5), camera=(iter, 30))
    
    # Bottom left: Phase plane
    plot!(plt[3], model.data[1,:], model.data[2,:], 
          label="True Solution", lw=2, color=:black,
          xlabel="x", ylabel="y",
          xlim=(0,4.5),ylim=(0,4.5),
           title="Phase Plane")
    plot!(plt[3], nn_sol[1,:], nn_sol[2,:], 
          label="Prediction", lw=2, color=:red, linestyle=:dash)
    scatter!(plt[3], [model.data[1,1]], [model.data[2,1]], color=:blue, label="Initial Condition")
    
    # Bottom right: Time series comparison
    plot!(plt[4], model.times, model.data[1,:], label="True x", color=:blue, lw=2)
    plot!(plt[4], model.times, model.data[2,:], label="True y", color=:green, lw=2)
    plot!(plt[4], model.times, nn_sol[1,:], label="KAN x", color=:blue, linestyle=:dash, lw=2)
    plot!(plt[4], model.times, nn_sol[2,:], label="KAN y", color=:green, linestyle=:dash, lw=2)
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

##Ground Truth Data Generation ########
function lotka!(du,u,p,t) 
    du[1] = p.r*u[1] - 0.5f0*u[1]*u[2]
    du[2] = 0.5f0*u[1]*u[2] - p.m*u[2]
end

#data generation parameters
dt=1.0
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
#model = CustomDerivatives(data, lotka_voltera!, initial_parameters)
model = CustomDerivatives(data, dudt!, initial_parameters)
##Training
N_iter = 100
iterator = ProgressBar(1:N_iter)
iters_per_loop = 10
@time for i in 1:N_iter#iterator

    #Update model 
    # set optimization problem 
    target  = (x,p) -> model.loss_function(x)
    adtype  = Optimization.AutoZygote()
    optf    = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, model.parameters)

    # run optimizer for 10 steps
    @time sol = Optimization.solve(optprob, 
                                Optimisers.Adam(0.05), 
                                maxiters = iters_per_loop)
    
    # assign parameters to model 
    model.parameters = sol.u

    # CALLBACK
    loss_curr=deepcopy(model.loss_function(model.parameters))
    #loss_curr_test=deepcopy(loss_test(p))
    #@time set_description(iterator, string("Iter:", i, "| Loss:", @sprintf("%.2f", loss_curr), "|"))

    #display(save_training_frame(model, pM, i, "frames_2"; save=false))
    print("Hello?!?!?")
    print(pM)
    display(UniversalDiffEq.plot_state_estimates(model))
    #display(UniversalDiffEq.plot_predictions(model))
    # SAVE
    #callback(i)
end