using Printf
# This is just to suppress warnings
using Suppressor

const COLORS = (
    test_loss = :tomato,
    loss = :royalblue1,
    timeseries_x = :lightskyblue2,
    timeseries_y = :royalblue1,
    true_solution = :black,
    prediction = :royalblue1,
    initial_condition = :blue,
    training_region = :gray
)
dir = @__DIR__
# Structure for passing data for the 2d plot which doesn't need to be recalculated
struct StaticData_2D
    x::StepRangeLen
    y::StepRangeLen
    xy::Matrix{Tuple{Float64,Float64}}
    true_h::Matrix{Float64}
    tspan_train::Tuple{Float32,Float32}
    u0::Vector{Float32}
    spinning_rate::Float32
end

# Structure for passing data for the 1d plot which doesn't need to be recalculated
struct StaticData_1D
    x::StepRangeLen
    true_h::Vector{Float64}
    tspan_train::Tuple{Float32,Float32}
    u0::Vector{Float32}
    spinning_rate::Float32
    sol_max_x::Float32
end

function find_frame_directory()
    # Create directories if needed
    folder_count = 0
    training_dir = joinpath(dir, "tests", "training_frames_$folder_count")
    # Set up the folder if it doesn't already exist
    if !isdir(joinpath(dir, "tests"))
        println("Making test folder")
        mkdir(joinpath(dir, "tests"))
    end
    if !isdir(training_dir)
        println("Making training_frames_$folder_count folder")
        mkdir(training_dir)
    end
    is_empty = isempty(readdir(training_dir))
    # If it's already a directory and it's not empty, try the next folder
    while isdir(training_dir) && !is_empty 
        folder_count += 1 
        training_dir = joinpath(dir, "tests", "training_frames_$folder_count")
    end
    if !is_empty
        print("Making directory ", string(training_dir))
        mkdir(training_dir)
    end
    training_dir
end

function plot_loss(plt, loss, test_loss, iter)
    plot!(plt, loss, color = COLORS.loss, label = "training loss",
          title = string("Loss Profiles| Loss:", @sprintf("%.4e", loss[end]), "|",
                          "Test_Loss:", @sprintf("%.4e", test_loss[end])), xlabel = "iteration",
          yaxis = :log)
    # plot!(plt, test_loss, color = COLORS.test_loss, label = "test loss")
end

function plot_phase_plane(plt, model, UDE_sol, static_data)
    plot!(plt, model.data[1,:], model.data[2,:], 
          label = "True Solution", lw = 2, color = COLORS.true_solution,
          xlabel = "x", ylabel = "y",
          xlim = (0, 4.5), ylim = (0, 4.5),
          title = "Phase Plane")
    plot!(plt, UDE_sol[:,2], UDE_sol[:,3], 
          label = "KAN Prediction", lw = 2, color = COLORS.prediction, linestyle = :dash)
    scatter!(plt, [static_data.u0[1]], [static_data.u0[2]], color = COLORS.initial_condition, label = "Initial Condition")
end

function plot_time_series(plt, model, UDE_sol, static_data)
    plot!(plt, model.times, model.data[1,:], label = "True x", color = COLORS.timeseries_x, lw = 2)
    plot!(plt, model.times, model.data[2,:], label = "True y", color = COLORS.timeseries_y, lw = 2)
    plot!(plt, UDE_sol[:,1], UDE_sol[:,2], label = "UDE Solution w/ KAN x", color = COLORS.timeseries_x, linestyle = :dash, lw = 2)
    plot!(plt, UDE_sol[:,1], UDE_sol[:,3], label = "UDE Solution w/ KAN y", color = COLORS.timeseries_y, linestyle = :dash, lw = 2)
    vspan!(plt, [static_data.tspan_train[1], static_data.tspan_train[2]], alpha = 0.2, color = COLORS.training_region, label = "Training Region",
           xlabel = "Time", ylabel = "Population",
           ylim = (0, 6),
           title = "Time Series Comparison", legend = :topright)
end

function save_training_frame_2d(static_data::StaticData_2D, model::UDE, nn, pM, iter, loss, test_loss, training_dir; save = false)
    @suppress begin
        # KAN prediction
        nn_h = [nn([i, j], pM, stM)[1][] for (i, j) in static_data.xy]

        # Get predictions for trajectories
        UDE_sol = UniversalDiffEq.forecast(model, static_data.u0, model.times)
        # Create 2x2 grid plot
        plt = plot(layout = (2, 2), size = (1600, 1200), titlefontsize = 12)

        # Top row: Interaction function surfaces
        plot!(plt[1], static_data.x, static_data.y, static_data.true_h, st = :surface, 
              title = "True Interaction h(x,y) compared with KAN (Iteration $iter)",
              xlabel = "x", ylabel = "y", zlabel = "h(x,y)",
              alpha = 0.4, c = :blues,
              camera = (iter * static_data.spinning_rate, 30))
        
        plot!(plt[1], static_data.x, static_data.y, nn_h, st = :surface, c = :blues,
              alpha = 0.4, label = "KAN")
    
        # Top right: loss
        plot_loss(plt[2], loss, test_loss, iter)

        # Bottom left: Phase plane
        plot_phase_plane(plt[3], model, UDE_sol, static_data)
    
        # Bottom right: Time series comparison
        plot_time_series(plt[4], model, UDE_sol, static_data)

        if save
            # Save figure with iteration number
            savefig(plt, joinpath(training_dir, "frame_$(lpad(iter, 5, '0')).png"))
        end
    
        return plt
    end
end

function save_training_frame_1d(static_data::StaticData_1D, model::UDE, nn, pM, iter, loss, test_loss, training_dir; save = false)
    # KAN prediction
    nn_h = [nn([i], pM, stM)[1][] for (i) in static_data.x]

    # Get predictions for trajectories
    UDE_sol = UniversalDiffEq.forecast(model, static_data.u0, model.times)
    # Create 2x2 grid plot
    plt = plot(layout = (2, 2), size = (1600, 1200), titlefontsize = 12)
    
    x = static_data.x
    # Top row: Interaction function surfaces
    plot!(plt[1], x, x .* (1 .- x),
          title = "True Interaction: h(x) = x(1-x)/10",
          xlabel = "x", ylabel = "y",
          label = "True h(x)",
          color = :black,
          xlim = (-1, static_data.sol_max_x + 1))
    plot!(plt[1], x, nn_h, 
          title = "KAN Learned Interaction (Iteration $iter)",
          label = "KAN(x,θ)",
          color = :red,
          linestyle = :dash)
    vspan!(plt[1], [0,static_data.sol_max_x], alpha = 0.2, color = :gray, label = "State Space")

    # Top right: loss
    plot_loss(plt[2], loss, test_loss, iter)

    # Bottom left: Phase plane
    plot_phase_plane(plt[3], model, UDE_sol, static_data)
    
    # Bottom right: Time series comparison
    plot_time_series(plt[4], model, UDE_sol, static_data)

    if save
        # Save figure with iteration number
        savefig(plt, joinpath(training_dir, "frame_$(lpad(iter, 5, '0')).png"))
    end
    return plt
end
