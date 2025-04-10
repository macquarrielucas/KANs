#This code is for training an MLPs 
## PACKAGES AND INCLUSIONS
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
using JLD2
using Logging
using ForwardDiff
#My packages and functions
include("../helpers/plotting_functions.jl")
include("../helpers/helpers.jl")
include("../helpers/loss_functions.jl")
include("../helpers/KolmogorovArnold.jl")
using .KolmogorovArnold

function h(x)
    #The hidden function that we are trying to learn
    x*(1-x)
end

function lotka!(du,u,p,t) 
    du[1] = h(u[1])- 0.5*u[1]*u[2]
    du[2] = 0.5*u[1]*u[2] - 0.03*u[2]
end

# CONSTRUCT KAN-ODES
function ude!(model, du, u, p, stM, t)
    NN_(x) = model([x], p, stM)[1][1]
    du[1] = NN_(u[1]) - 0.5*u[1]*u[2]
    du[2] = 0.5*u[1]*u[2] - 0.03*u[2]
end

function generate_data()
    #data generation parameters
    dt::Float32 =0.1f0
    tspan_test::Tuple{Float32,Float32} = (0.0, 500)
    tspan_train::Tuple{Float32,Float32} =(0.0, 100)
    u0::Vector{Float32} = [1.0f0, 1.0f0]
    p_=Float32[] #These can only hold an array of parameters which are floats.
    prob = ODEProblem(lotka!, u0,tspan_test,p_)

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

function define_MLP(rng)
    # Define MLP
    layer_width::Int=10
    mlp = Lux.Chain(
        Lux.Dense(1 => 10, relu),
        Lux.Dense(10 => 10, relu),
        Lux.Dense(10 => 1),
        )
    pM , stM  = Lux.setup(rng, mlp)
    return mlp, pM, stM
end

function train_mlp(N_iter::Int; 
    SAVE_MODEL_ON::Bool = true)

    dir = @__DIR__
    training_dir = get_training_dir(dir)
    @info "Training directory: $training_dir"
    #Random
    rng = Random.default_rng()
    Random.seed!(rng, 3)
    @info "Generating data..."
    t_test, Xn_test, t_train, Xn_train = generate_data()
    u0 = Xn_test[:,1]
    
    @info "Setting up MLP..."
    mlp, pM, stM = define_MLP(rng)
    pM_data     = getdata(ComponentArray(pM))
    pM_axis     = getaxes(ComponentArray(pM))
    p = ComponentArray(pM_data, pM_axis) 
    #This is needed to pass onto training step
    function UDE1!(du,u,p,t)
        ude!(mlp, du, u, p, stM, t)
    end
    
    #opt = Flux.Momentum(1e-3, 0.9)
    opt = Flux.Adam(1e-4)
    pred_length::Int = 100
    REGULARIZATION = 1
    REG_COEFF=1e-2
    #Things to include in the plot
    hyperparameter_string = [
        "dt: 0.1, tspan_train: $(t_train[1]) to $(t_train[end]), tspan_test: $(t_test[1]) to $(t_test[end])",
        "u0: $(u0[1]) $(u0[2])",
        "Training iters: $N_iter",
        "optimizer: Adam(1e-4)",
        #"layer_width: $layer_width, grid_size: $grid_size",
        "Architecture: [1,3,5],[3,3,5],[3,1,5]",
        "Number of parameters: $(length(p))",
        "basis_func: rbf, normalizer: softsign",
        "Loss type: multiple_shooting_loss", 
        "regularization : $REGULARIZATION",
        "regularization coeffecient: $REG_COEFF",
        "Prediction length for multiple shooting: $pred_length",
        "Save at: $training_dir",
        "ODE: x' =  h(x,y) - 0.5xy \n y' = 0.5xy -0.03y",
        "h(x,y)=0.5xy"
    ]

    l1 = Real[]
    l1_test=Real[] 
    iterator = ProgressBar(1:N_iter)
    @info "Beginning training loop..."
    for i in iterator
        # GRADIENT COMPUTATION
        #println("Computing gradient... ($i)")
        #I think theres a way to get the loss in the call, instead of calling it again for loss_curr
        grad = ForwardDiff.gradient(p -> loss_train(UDE1!, p, t_train, Xn_train; sparse_on=REGULARIZATION, reg_coeff=REG_COEFF, pred_length=pred_length), p)

        # UPDATE WITH ADAM OPTIMIZER
        update!(opt, p, grad)

        
        #Add loss to the lists 
        append!(l1, loss_train(UDE1!, p,t_train, Xn_train;sparse_on=REGULARIZATION,  reg_coeff=REG_COEFF, pred_length=pred_length))
        append!(l1_test, loss_train(UDE1!, p, t_test, Xn_test;sparse_on=REGULARIZATION, reg_coeff=REG_COEFF, pred_length=pred_length))
        #append!(p_list, [deepcopy(p)])
        #Update visuals
        set_description(iterator, string(
            "Iter:", i, 
            "| Loss:", @sprintf("%.2e", l1[end]), 
            "| Test_Loss:", @sprintf("%.2e", l1_test[end]), 
            "|"
        ))
        if i%1000==0 & SAVE_MODEL_ON
            save_model_parameters(i,N_iter,p ,stM,training_dir, "MLP")
        end
    end
    @info "Training complete. Saving model parameters..."
    save_model_parameters(N_iter,N_iter,p ,stM,training_dir, "MLP")
    @info "Saving loss profiles..."
    @save joinpath(training_dir, "final_losses_MLP.jld2") l1 l1_test
end
#main(100; SAVE_PLOTS_ON=false, SAVE_MODEL_ON=true, DISPLAY=false)