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
#My packages and functions
include("../helpers/plotting_functions.jl")
include("loss_functions.jl") #this is not implemented yet
# KAN PACKAGE LOAD
include("../Lotka-Volterra/src/KolmogorovArnold.jl")
using .KolmogorovArnold

function h(x,y)
    #The hidden function that we are trying to learn
    0.5*x*y
end

function lotka!(du,u,p,t) 
    du[1] = u[1]*(1-u[1])- h(u[1],u[2])
    du[2] = h(u[1],u[2]) - u[2]
end

function kanode!(model, du, u, p, stM, t)
    kan1_(x) = model(x, p, stM)[1][1]
    du[1] =  u[1]*(1-u[1]) - kan1_(u)
    du[2] = kan1_(u) - u[2]
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
    layer_width::Int=5
    grid_size::Int=5
    #This KAN looks like
    # psi(normalizer(phi1(normalizer(x)) + phi2(normalizer(x)) + phi3(normalizer(x)) + ... + phi10(normalizer(x))))
    kan = Lux.Chain(
        KolmogorovArnold.KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
        KolmogorovArnold.KDense(layer_width,  1, grid_size; use_base_act = true, basis_func, normalizer),
    )
    pM , stM  = Lux.setup(rng, kan)
    return kan, pM, stM
end

function main()
    #Random
    rng = Random.default_rng()
    Random.seed!(rng, 3)
    println("Generating data...")
    t_test, Xn_test, t_train, Xn_train = generate_data()
    u0 = Xn_test[:,1]
    println("Initializing KAN...")
    kan1, pM, stM = define_KAN(rng)
    pM_data     = getdata(ComponentArray(pM))
    pM_axis     = getaxes(ComponentArray(pM))
    p = ComponentArray(pM_data, pM_axis) 
    #p = (deepcopy(pM_data))./1e5 ; #this was in the original code, but i dont know why

    println("Defining static data...")
    ##DATA FOR PLOTTING
    # Create grid for interaction function visualization
    observation_data = [t_test Xn_test'] #This is the data that we will be comparing to
    x = range(0, 1, length=20)
    y = range(0, 1, length=20)
    xy = [(i,j) for i in x, j in y]
    # True interaction function   
    tspan_train = (t_train[1], t_train[end])
    true_h = [h(i,j) for (i,j) in xy]
    spinning_rate = 0.2
    static_data = StaticData_2D(
                    observation_data,
                    x,
                    y,
                    xy,
                    true_h,
                    tspan_train,
                    u0,
                    spinning_rate)

    #This is needed to pass onto training step
    function UDE!(du,u,p,t)
        kanode!(kan1, du, u, p, stM, t)
    end
                            
    SAVE_ON::Bool = false
    if SAVE_ON
        dir = @__DIR__
        training_dir = find_frame_directory(dir)
        println("Saving frames to: ", training_dir)
    else
        training_dir=""
    end
    #opt = Flux.Momentum(1e-3, 0.9)
    opt = Flux.Adam(1e-4)
    N_iter::Int = 10000
    iterator = ProgressBar(1:N_iter)

    #Stuff to track loss and test loss
    l = Real[]
    l_test=Real[]
    #p_list = []
    println("Training...")
    for i in iterator
        # GRADIENT COMPUTATION
        #println("Computing gradient... ($i)")
        #I think theres a way to get the loss in the call, instead of calling it again for loss_curr
        grad = Zygote.gradient(p -> loss_train(UDE!, p, t_train, Xn_train), p)[1]

        # UPDATE WITH ADAM OPTIMIZER
        update!(opt, p, grad)

        #Add loss to the lists 
        append!(l, loss_train(UDE!, p,t_train, Xn_train))
        append!(l_test, loss_train(UDE!, p, t_test, Xn_test))
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
            #UDE_forecast = multiple_shooting_predict(UDE!, p, 10, t_test, Xn_test)
            UDE_forecast = single_shooting_predict(UDE!, p, Xn_test[:,1], t_test)
            #UDE_forecast = single_shooting_predict(UDE!, p,u0, t_test)
            UDE_sol=[t_test UDE_forecast']
            #println("Plotting...")
            display(save_training_frame(static_data, UDE_sol, kan1,p, stM, i, l,l_test,training_dir; save=SAVE_ON))
        end
    end
end
main()