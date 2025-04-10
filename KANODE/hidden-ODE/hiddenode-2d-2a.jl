# PACKAGES AND INCLUSIONS
using Profile 
using ChainRulesCore
using ComponentArrays
using ComponentArrays: getaxes
using ComponentArrays: getdata
using DiffEqFlux
using Flux
using Flux: mae, mean, update!
using OrdinaryDiffEq
using Plots
using Printf
using ProgressBars
using Random
using Zygote
using Lux
using Logging

#My packages and functions
include("../helpers/plotting_functions.jl")
include("../helpers/helpers.jl")
include("../helpers/loss_functions.jl")
include("../helpers/KolmogorovArnold.jl")
using .KolmogorovArnold

function h(x,y)
    #The hidden function that we are trying to learn
    0.5*x*y
end

function sir!(du,u,p,t) 
    γ=1.0f0
    S, I, R = u[1], u[2], u[3]
    du[1] = - h(S,I)
    du[2] = h(S,I) - γ*I
    du[3] = γ*I
end
#test git 
function ude!(model, du, u, p, stM, t)
    NN_(x) = model(x, p, stM)[1]
    S, I, R = u[1], u[2], u[3]
    du[1] = -  NN_(x)
    du[2] =  NN_(x) - γ*I
    du[3] = γ*I
end

function generate_data()
    #data generation parameters
    dt::Float32 =0.01f0
    tspan_test::Tuple{Float32,Float32} = (0.0, 100)
    tspan_train::Tuple{Float32,Float32} =(0.0, 10)
    u0::Vector{Float32} = [1.0f0, 0.1f0, 0.0f0]
    p_=Float32[] #These can only hold an array of parameters which are floats.
    prob = ODEProblem(sir!, u0,tspan_test,p_)

    #generate training data, split into train/test
    solution = solve(prob, Tsit5(), abstol = 1e-6, reltol = 1e-8, saveat = dt)
    # Calculate the index to split the solution into training and test sets based on the training time span
    end_index = Int64(floor(length(solution.t) * tspan_train[2] / tspan_test[2])) + 1
    t_test = solution.t #full dataset
    t_train=t_test[1:end_index] #training cut
    #####
    Xn_test = Array(solution)
    Xn_train = Xn_test[:, 1:end_index]
    return t_test, Xn_test, t_train, Xn_train
end
function defineMLP(rng)
    # Define the MLP architecture
    mlp = Lux.Chain(
        Lux.Dense(2, 5, Lux.relu),
        Lux.Dense(5, 5, Lux.relu),
        Lux.Dense(5, 1)
    )
    pM , stM  = Lux.setup(rng, mlp)
    return mlp, pM, stM
end

# Configure logging
global_logger(ConsoleLogger(stderr, Logging.Info))

function main()
    #Random
    rng = Random.default_rng()
    Random.seed!(rng, 3)
    @info "Generating data..."
    t_test, Xn_test, t_train, Xn_train = generate_data()
    @info "Setting initial condition for training"
    u0 = Xn_test[:,1]
    @info "Initializing MLP..."
    mlp, pM, stM = defineMLP(rng)
    pM_data     = getdata(ComponentArray(pM))
    pM_axis     = getaxes(ComponentArray(pM))
    p = ComponentArray(pM_data, pM_axis) 
    #p = (deepcopy(pM_data))./1e5 ; #this was in the original code, but i dont know why

    @info "Defining static data..."
    ##DATA FOR PLOTTING
    # Create grid for interaction function visualization
    observation_data = [t_test Xn_test'] #This is the data that we will be comparing to
    x = range(0, 3, length=20)
    y = range(0, 3, length=20)
    xy = [(i,j) for i in x, j in y]
    # True interaction function   
    tspan_train = (t_train[1], t_train[end])
    true_h = [h(i,j) for (i,j) in xy]
    obs_curve = [h(i,j) for (i,j) in zip(Xn_test[1,:], Xn_test[2,:])]
    spinning_rate = 0.2
    static_data = StaticData_2D(
                    observation_data,
                    x,
                    y,
                    xy,
                    true_h,
                    tspan_train,
                    u0,
                    spinning_rate,
                    obs_curve)

    #This is needed to pass onto training step
    function UDE!(du,u,p,t)
        ude!(mlp, du, u, p, stM, t)
    end
            
    ##Settings
    SAVE_ON::Bool = true

    dir = @__DIR__
    training_dir = get_training_dir(dir)
    #opt = Flux.Momentum( 1e-3, 0.9)
    opt = Flux.Adam(1e-4)
    N_iter::Int = 10000
    iterator = ProgressBar(1:N_iter)
    pred_length::Int = 100
    #Things to include in the plot
    hyperparameter_string = [
        "N_iter: $N_iter",
        "dt: 0.01",
        "tspan_train: $(t_train[1]) to $(t_train[end])",
        "tspan_test: $(t_test[1]) to $(t_test[end])",
        "u0: $(u0[1]) $(u0[2])",
        "Training iters: $N_iter",
        "optimizer: Adam(1e-4)",
        "spinning_rate: 0.2",
        "Loss type: single_shooting_loss",
        "SAVE_ON: $SAVE_ON",
        "ODE: S' =  - h(S,I) \n I' = h(S,I)-γ*I \n R' = γ*I",
        "h(S,i)=0.5SI"
    ]
    #Stuff to track loss and test loss
    l = Real[]
    l_test=Real[]
    #p_list = []
    @info "Training..."
    for i in iterator
        # GRADIENT COMPUTATION
        #println("Computing gradient... ($i)")
        #I think theres a way to get the loss in the call, instead of calling it again for loss_curr
        grad = Zygote.gradient(p -> loss_train(UDE!, p, t_train, Xn_train), p)[1]

        # UPDATE WITH ADAM OPTIMIZER
        update!(opt, p, grad)

        #Add loss to the lists 
        append!(l, loss_train(UDE!, p,t_train, Xn_train;pred_length= pred_length))
        append!(l_test, loss_train(UDE!, p, t_test, Xn_test;pred_length= pred_length))
        #append!(p_list, [deepcopy(p)])

        #Update visuals
        set_description(iterator, string(
            "Iter:", i, 
            "| Loss:", @sprintf("%.2e", l[end]), 
            "| Test_Loss:", @sprintf("%.2e", l_test[end]), 
            "|"
        ))
        if i%10 == 0 || i==1
            #Turn the data into an nx3 matrix for the plotting function
            #UDE_forecast = multiple_shooting_predict(UDE!, p, pred_length, t_test, Xn_test)
            #UDE_forecast = single_shooting_predict(UDE!, p, Xn_test[:,1], t_test)
            UDE_forecast = single_shooting_predict(UDE!, p,u0, t_test)
            UDE_sol=[t_test UDE_forecast']
            #println("Plotting...")
            display(plot_training_frame(static_data, UDE_sol, mlp,p, stM, i, l,l_test,hyperparameter_string))
        end
    end
end
main()