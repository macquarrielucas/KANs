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
function kanode!(model, du, u, p, stM, t)
    kan1_(x) = model([x], p, stM)[1][1]
    du[1] = kan1_(u[1]) - 0.5*u[1]*u[2]
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
"""
    Defines the KAN and returns the model, parameters and state.

# Arguments
- `rng`: A random number generator used for initializing the model parameters.

# Returns
- `kan`: The defined Kolmogorov-Arnold Network (KAN) model.
- `pM`: The parameters of the KAN model.
- `stM`: The state of the KAN model.

# Description
This function defines a Kolmogorov-Arnold Network (KAN) using radial basis functions (rbf) and a softsign normalizer. The network consists of two layers with a specified layer width and grid size. The function initializes the model parameters and state using the provided random number generator and returns them along with the model.
"""
function define_KAN(rng)
    # Define KAN
    basis_func = KolmogorovArnold.rbf      # rbf, rswaf
    normalizer = KolmogorovArnold.softsign # sigmoid(_fast), tanh(_fast), softsign
    layer_width::Int=3
    grid_size::Int=5
    #This KAN looks like
    # psi(normalizer(phi1(normalizer(x)) + phi2(normalizer(x)) + phi3(normalizer(x)) + ... + phi10(normalizer(x))))
    kan = Lux.Chain(
        KolmogorovArnold.KDense( 1, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
        KolmogorovArnold.KDense(layer_width,  1, grid_size; use_base_act = true, basis_func, normalizer),
    )
    pM , stM  = Lux.setup(rng, kan)
    return kan, pM, stM, layer_width, grid_size
end

function main(N_iter::Int;
    SAVE_PLOTS_ON::Bool = true, 
    SAVE_MODEL_ON::Bool = true,
    DISPLAY::Bool = true)

    dir = @__DIR__
    training_dir = get_training_dir(dir)
    #Random
    rng = Random.default_rng()
    Random.seed!(rng, 3)
    println("Generating data...")
    t_test, Xn_test, t_train, Xn_train = generate_data()
    u0 = Xn_test[:,1]
    println("Initializing KAN...")
    kan1, pM, stM, layer_width, grid_size = define_KAN(rng)
    pM_data     = getdata(ComponentArray(pM))
    pM_axis     = getaxes(ComponentArray(pM))
    p = ComponentArray(pM_data, pM_axis) 
    #p = (deepcopy(pM_data))./1e5 ; #this was in the original code, but i dont know why

    println("Defining static data...")
    ##Plotting
    sol_max_x =maximum(Xn_train[1,:])  #Bounds on the interaction plot
    x = range(-1, sol_max_x+1, length=40)
    true_h = [h(i) for i in x]  # True interaction function   
    tspan_train = (t_train[1], t_train[end])
    observation_data = [t_test Xn_test'] #This is the data that we will be comparing to
    static_data = StaticData_1D(observation_data, 
                                x,
                                true_h,
                                tspan_train, 
                                u0, 
                                sol_max_x)

    #This is needed to pass onto training step
    function UDE!(du,u,p,t)
        kanode!(kan1, du, u, p, stM, t)
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
    #Stuff to track loss and test loss
    l = Real[]
    l_test=Real[]
    #p_list = []

    println("Training...")

    iterator = ProgressBar(1:N_iter)
    for i in iterator
        # GRADIENT COMPUTATION
        #println("Computing gradient... ($i)")
        #I think theres a way to get the loss in the call, instead of calling it again for loss_curr
        grad = Zygote.gradient(p -> loss_train(UDE!, p, t_train, Xn_train;sparse_on=REGULARIZATION, reg_coeff=REG_COEFF,pred_length=pred_length), p)[1]

        # UPDATE WITH ADAM OPTIMIZER
        update!(opt, p, grad)

        
        #Add loss to the lists 
        append!(l, loss_train(UDE!, p,t_train, Xn_train;sparse_on=REGULARIZATION,  reg_coeff=REG_COEFF, pred_length=pred_length))
        append!(l_test, loss_train(UDE!, p, t_test, Xn_test;sparse_on=REGULARIZATION, reg_coeff=REG_COEFF, pred_length=pred_length))
        #append!(p_list, [deepcopy(p)])
        #=
        #Update visuals
        set_description(iterator, string(
            "Iter:", i, 
            "| Loss:", @sprintf("%.2e", l[end]), 
            "| Test_Loss:", @sprintf("%.2e", l_test[end]), 
            "|"
        ))
            =#
        if (i % 10 == 0 || i == 1) && (SAVE_PLOTS_ON || DISPLAY)
            plot1 = plot_KAN_diagram(kan1, p::ComponentArray, stM, reshape(Xn_train[1,:], 1, :))

            # Turn the data into an nx3 matrix for the plotting function
            UDE_forecast = multiple_shooting_predict(UDE!, p, pred_length, t_test, Xn_test)
            # UDE_forecast = single_shooting_predict(UDE!, p, Xn_test[:, 1], t_test)
            # UDE_forecast = single_shooting_predict(UDE!, p, u0, t_test)
            UDE_sol = [t_test UDE_forecast']
            # println("Plotting...")
            plot2 = plot_training_frame(static_data, UDE_sol, kan1, p, stM, i, l, l_test, hyperparameter_string)
            plt = plot(plot2, plot1, layout = @layout([a; b]))
    
            if SAVE_PLOTS_ON
                # Save figure with iteration number
                if !isdir(joinpath(training_dir,"frames"))
                    mkdir(joinpath(training_dir,"frames"))
                end
                savefig(plt, joinpath(training_dir,"frames", "frame_$(lpad(i, 5, '0')).png"))
            end
            if DISPLAY
                display(plt)     
            end
        end
        if i % 1000 == 0 && SAVE_MODEL_ON
            save_model_parameters(i,N_iter,p ,stM,training_dir)
        end
    end
end
#main(25000; SAVE_PLOTS_ON=true, SAVE_MODEL_ON=true)