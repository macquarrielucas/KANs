using Lux, LuxCore
using Flux: mae, update!, mean
using Random
using Zygote 
using OrdinaryDiffEq
using UniversalDiffEq, DataFrames
using ProgressBars, Printf, Plots
using Optimization, Optimisers, OptimizationOptimJL
using Parameters
include("../Lotka-Volterra/src/KolmogorovArnold.jl")
using .KolmogorovArnold
include("../helpers/plotting_functions.jl")

# --------------------------
# GLOBAL CONSTANTS & COLORS
# --------------------------


# --------------------------
# PARAMETERS STRUCTURE
# --------------------------
@with_kw struct TrainingParams
    # Simulation parameters
    dt::Float32 = 0.1f0
    tspan::Tuple{Float32,Float32} = (0.0f0, 14.0f0)
    u0::Vector{Float32} = [1.0f0, 1.0f0]
    true_params::NamedTuple = (r=1.0f0, m=1.0f0)
    
    # Model architecture
    basis_func::Function = KolmogorovArnold.rbf
    normalizer::Function = softsign
    num_layers::Int = 2
    layer_width::Int = 7
    grid_size::Int = 5
    
    # Training configuration
    max_iters::Int = 100
    learning_rate::Float32 = 0.05f0
    optimizer::Symbol = :Adam  # Options: :Adam, :LBFGS, :RMSProp
    iters_per_loop::Int = 5
    loss_log_interval::Int = 5
    
    # Visualization parameters
    spinning_rate::Float32 = 0.5f0
    plot_range::Tuple{Float32,Float32} = (0.0f0, 3.0f0)
    plot_resolution::Int = 20
    SAVE_ON::Bool = false
end

# --------------------------
# MODEL DEFINITIONS
# --------------------------
function lotka!(du, u, p, t) 
    du[1] = p.r * u[1] - 0.5f0 * u[1] * u[2]
    du[2] = 0.5f0 * u[1] * u[2] - p.m * u[2]
end

function create_kan_model(params::TrainingParams)
    Lux.Chain(
        KDense(2, params.layer_width, params.grid_size; 
              use_base_act=true, params.basis_func, params.normalizer),
        KDense(params.layer_width, 1, params.grid_size;
              use_base_act=true, params.basis_func, params.normalizer),
    )
end

# --------------------------
# DATA MANAGEMENT
# --------------------------

function generate_static_data(params::TrainingParams)
    x = range(params.plot_range..., length=params.plot_resolution)
    y = range(params.plot_range..., length=params.plot_resolution)
    xy = [(i,j) for i in x, j in y]
    true_h = [0.5*i*j for (i,j) in xy]
    
    StaticData(x, y, xy, true_h, params.tspan, params.u0, params.spinning_rate)
end

# --------------------------
# TRAINING LOGIC
# --------------------------
function get_optimizer(params::TrainingParams)
    if params.optimizer == :Adam
        return Optimisers.Adam(params.learning_rate)
    elseif params.optimizer == :LBFGS
        return Optim.LBFGS()
    elseif params.optimizer == :RMSProp
        return Optimisers.RMSProp(params.learning_rate)
    else
        error("Unsupported optimizer: $(params.optimizer)")
    end
end

function train_model!(model, static_data, params::TrainingParams)
    iterator = ProgressBar(1:params.max_iters)
    loss_history = Float32[]
    loss_test_history = Float32[]
    opt = get_optimizer(params)

    for i in iterator
        # Optimization setup
        optprob = Optimization.OptimizationProblem(
            (x,p) -> model.loss_function(x),
            model.parameters
        )

        # Parameter update
        sol = Optimization.solve(optprob, opt, maxiters=params.iters_per_loop+1)
        model.parameters = sol.u

        # Logging
        current_loss = model.loss_function(model.parameters)
        push!(loss_history, current_loss)

        set_description(iterator, "Iter: $i | Loss: $(@sprintf("%.2e", current_loss))")
        # Visualization
        if i % params.loss_log_interval == 0
            @time begin
                display(save_training_frame_2d(static_data, model, params, i, loss_history. loss_test_history,training_dir; save=SAVE_ON))
            end
        end
    end
    
    loss_history
end

# --------------------------
# MAIN EXECUTION
# --------------------------
function main(params::TrainingParams)
    # Initialize components
    data = generate_training_data(params)
    kan_model = create_kan_model(params)
    rng = Random.default_rng()
    Random.seed!(rng, 3)
    pM, stM = Lux.setup(rng,kan1)
    
    # Create UDE model
    initial_params = (NN=pM, r=2.0f0, m=0.5f0)
    ude_model = CustomDerivatives(data, lotka_voltera!, initial_params)

    # Prepare visualization data
    static_data = generate_static_data(params)

    # Run training
    train_model!(ude_model, static_data, params)
end

# --------------------------
# PARAMETER EXPORT
# --------------------------
function save_params(params::TrainingParams, filename::String)
    open(filename, "w") do io
        println(io, "# Training Parameters")
        for field in fieldnames(TrainingParams)
            val = getfield(params, field)
            println(io, "$field = $val")
        end
    end
end

# Example usage:
if abspath(PROGRAM_FILE) == @__FILE__
    params = TrainingParams(
        optimizer=:Adam,
        max_iters=100,
        SAVE_ON=false
    )
    main(params)
    save_params(params, "training_parameters.cfg")
end