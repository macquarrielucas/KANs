using Printf
using Suppressor
using Plots
include("helpers.jl")
const COLORS = (
    #Hidden function Plot (Top left)
    pred_hidden_function=:red,
    true_hidden_function=:black,
    #Loss (Top right)
    test_loss = :tomato,
    loss = :royalblue1,
    #Phaseplane     
    true_solution = :black,
    prediction = :royalblue1,
    train_prediction = :orange,
    initial_condition = :red,
    #Time series
    timeseries_x = :lightskyblue2,
    timeseries_y = :royalblue1,
    #etc 
    training_region = :gray
)
#

"""
Structure for passing data for the 2d plot which doesn't need to be recalculated
"""
struct StaticData_2D
    observation_data::Matrix #This contains the observations/testing data. 3xN matrix with time, x, y
    x::StepRangeLen # The x range that the interaction function will be plotted over
    y::StepRangeLen # The y range that the interaction function will be plotted over
    xy::Matrix{Tuple{Float64,Float64}} # The meshgrid of x and y values (should coincide with the ranges of x and y)
    true_h::Matrix{Float64} #The true interaction function to be found
    tspan_train::Tuple{Float32,Float32} #The time span of the training data
    u0::Vector{Float32} #Initial conditions for the system
    spinning_rate::Float32 #The rate at which the plot spins
    obs_curve::Vector{Float64} #The curve of the true interaction function over the observed data
end

"""
Structure for passing data for the 1d plot which doesn't need to be recalculated
"""
struct StaticData_1D
    observation_data::Matrix #This contains the observations/training data
    x::StepRangeLen
    true_h::Vector{Float64}
    tspan_train::Tuple{Float32,Float32}
    u0::Vector{Float32}
    sol_max_x::Float32
end
####################
## Subplots ##
####################
"""
    plot_loss(plt, loss::Vector{Real}, test_loss::Vector{Real}, iter::Int)::Nothing

Plots the training and test loss profiles on a given plot.

# Arguments
- `plt`: The plot object to which the loss profiles will be added.
- `loss::Vector{Real}`: A vector containing the training loss values.
- `test_loss::Vector{Real}`: A vector containing the test loss values.
- `iter::Int`: The current iteration number.

# Returns
- `Nothing`: This function does not return any value.

# Notes
- The function plots the training loss on a logarithmic y-axis.
- The test loss plot line is currently commented out.
"""
function plot_loss(plt, loss::Vector{Real}, test_loss::Vector{Real}, iter::Int)::Nothing
    plot!(plt, loss, color = COLORS.loss, label = "training loss",
          title = string("Loss Profiles (MSE)| Loss:", @sprintf("%.4e", loss[end]), "|",
                          "Test_Loss:", @sprintf("%.4e", test_loss[end])), xlabel = "iteration",
          yaxis = :log)
    if !isempty(test_loss)
        plot!(plt, test_loss, color = COLORS.test_loss, label = "test loss")
    end
    return nothing
end
"""
    plot_phase_plane(plt, UDE_sol::Matrix{Float64}, static_data::Union{StaticData_1D, StaticData_2D})::Nothing

Plot the phase plane of the true solution and the KAN prediction.

# Arguments
- `plt`: The plotting object to which the phase plane will be added.
- `UDE_sol::Matrix{Float64}`: The matrix containing the UDE solution with time in the first column, x-coordinates in the second column, and y-coordinates in the third column.
- `static_data::Union{StaticData_1D, StaticData_2D}`: The static data containing observation data, training span, and initial conditions.

# Details
- The function plots the true solution and the KAN prediction on the phase plane.
- It also highlights the solution over the training span and marks the initial condition.

# Returns
- `Nothing`: This function does not return any value.
"""
function plot_phase_plane(plt, UDE_sol::Matrix, static_data::Union{StaticData_1D, StaticData_2D})::Nothing
    obs_x = static_data.observation_data[:,2]
    obs_y = static_data.observation_data[:,3]
    obs_time = UDE_sol[:,1]
    pred_x = UDE_sol[:,2]
    pred_y = UDE_sol[:,3]
    # Plot the full solution
    plot!(plt, obs_x, obs_y, 
          label = "True Solution", lw = 2, color = COLORS.true_solution,
          xlabel = "x", ylabel = "y",
          #xlim = (0, 4.5), ylim = (0, 4.5), #this needs to be changed eventually
          title = "Phase Plane")
    plot!(plt, pred_x, pred_y, 
          label = "KAN Prediction", 
          lw = 2, color = COLORS.prediction, linestyle = :dash)
    
    # Plot the solution over the training span
    train_indices = findall(t -> t >= static_data.tspan_train[1] && t <= static_data.tspan_train[2], obs_time)
    plot!(plt, pred_x[train_indices], pred_y[train_indices], 
          label = "KAN Prediction (Training Span)", 
          lw = 2, color = COLORS.train_prediction, linestyle = :dash)#, zorder = 2)
    
    scatter!(plt, [static_data.u0[1]], [static_data.u0[2]], color = COLORS.initial_condition, label = "Initial Condition")#, zorder = 3)
    return nothing
end
"""
    plot_time_series(plt, UDE_sol::Matrix{Float64}, static_data::Union{StaticData_1D, StaticData_2D})::Nothing

Plots the time series data for both observed and predicted values.

# Arguments
- `plt`: The plot object to which the time series will be added.
- `UDE_sol::Matrix{Float64}`: A matrix containing the predicted time series data. The first column should be time, the second column should be the x values, and the third column should be the y values.
- `static_data::Union{StaticData_1D, StaticData_2D}`: An object containing the observed time series data and training time span. The `observation_data` field should be a matrix where the first column is time, the second column is the x values, and the third column is the y values. The `tspan_train` field should be a tuple containing the start and end times of the training period.

# Returns
- `Nothing`: This function modifies the plot object in place and does not return any value.
"""
function plot_time_series(plt, UDE_sol::Matrix, static_data::Union{StaticData_1D, StaticData_2D})::Nothing
    obs_times = static_data.observation_data[:,1]
    obs_x = static_data.observation_data[:,2]
    obs_y = static_data.observation_data[:,3]
    pred_times = UDE_sol[:,1]
    pred_x = UDE_sol[:,2]
    pred_y = UDE_sol[:,3]
    plot!(plt, obs_times, obs_x, label = "True x", color = COLORS.timeseries_x, lw = 2)
    plot!(plt, obs_times, obs_y, label = "True y", color = COLORS.timeseries_y, lw = 2)
    plot!(plt, pred_times, pred_x, label = "UDE Solution w/ KAN x", color = COLORS.timeseries_x, linestyle = :dash, lw = 2)
    plot!(plt, pred_times, pred_y, label = "UDE Solution w/ KAN y", color = COLORS.timeseries_y, linestyle = :dash, lw = 2)
    vspan!(plt, [static_data.tspan_train[1], static_data.tspan_train[2]], alpha = 0.2, color = COLORS.training_region, label = "Training Region",
           xlabel = "Time", ylabel = "Population",
           #ylim = (0, 6),
           title = "Time Series Comparison", legend = :topright)
    return nothing
end
"""
    plot_interaction_surface_2d(plt, static_data::StaticData_2D, nn_h, iter)::Nothing

Plots the interaction surface in 2D using the provided plotting object and data.

# Arguments
- `plt`: The plotting object to be used for plotting.
- `static_data::StaticData_2D`: An instance of `StaticData_2D` containing the x, y coordinates and true interaction values.
- `nn_h`: The predicted interaction values from the neural network.
- `iter`: The current iteration number, used for adjusting the camera angle.

# Returns
- `Nothing`: This function does not return any value.

# Description
This function plots the true interaction surface `h(x,y)` and the predicted interaction surface from the neural network on the same plot. The true interaction surface is plotted with a transparency of 0.4 and a blue color scheme. The camera angle is adjusted based on the iteration number to create a spinning effect.
"""
function plot_interaction_surface_2d(plt, static_data::StaticData_2D, nn_h, iter)::Nothing
    #=
    # Plot the true interaction surface
    plot!(plt, static_data.x, static_data.y, static_data.true_h, st = :surface, 
          title = "True Interaction h(x,y) compared with KAN (Iteration $iter)",
          xlabel = "x", ylabel = "y", zlabel = "h(x,y)",
          alpha = 0.4, c = :blues,
          camera = (iter * static_data.spinning_rate, 30),
          label = "True h(x,y)", colorbar = false)
    
    # Plot the predicted interaction surface
    plot!(plt, static_data.x, static_data.y, nn_h, st = :surface, c = :reds,
          alpha = 0.6, label = "KAN", colorbar = false)
    =#
    plot!(plt, static_data.x, static_data.y, abs.(nn_h-static_data.true_h), st = :surface, c = :reds,
            alpha = 0.6, label = "|KAN(x,y) - h(x,y)|", colorbar = false)
    # Plot the observation data as vertical lines
    obs_x = static_data.observation_data[:, 2]
    obs_y = static_data.observation_data[:, 3]
    #vspan!(plt, obs_x, obsy, alpha = 0.2, color = :gray, label = "Observation Data")
    plot!(plt, obs_x, obs_y, zeros(length(obs_x)), st = :path, label = "Observation Data", color = :black)
    
    # Add legend
    plot!(plt, legend = :topright)
    return nothing
end
"""
    plot_interaction_surface_1d(plt, static_data::StaticData_1D, nn_h, iter)

Plots the interaction surface for a 1D static data set.

# Arguments
- `plt`: The plot object to which the interaction surface will be added.
- `static_data::StaticData_1D`: A data structure containing the static data for the plot.
- `nn_h`: The neural network's learned interaction values.
- `iter`: The current iteration number, used for labeling the plot.

# Description
This function plots the true interaction surface `h(x) = x(1-x)/10` and the learned interaction surface from a neural network on the same plot. It also highlights the state space region.

# Plot Details
- The true interaction surface is plotted in black.
- The learned interaction surface is plotted in red with a dashed line.
- The state space region is highlighted with a gray vertical span.
"""
function plot_interaction_surface_1d(plt, static_data::StaticData_1D, nn_h, iter)
    plot!(plt, static_data.x, static_data.true_h,
          title = "True Interaction: h(x) = x(1-x)/10",
          xlabel = "x", ylabel = "y",
          label = "True h(x)",
          color = COLORS.true_hidden_function,
          xlim = (-1, static_data.sol_max_x + 1))
    plot!(plt, static_data.x, nn_h, 
          title = "KAN Learned Interaction (Iteration $iter)",
          label = "KAN(x,θ)",
          color = COLORS.pred_hidden_function,
          linestyle = :dash)
    vspan!(plt, [0,static_data.sol_max_x], alpha = 0.2, color = :gray, label = "State Space")
end


"""
    plot_hyperparameters(plt, hyperparams::Vector{String})::Nothing

Plots the hyperparameters in a text box.

# Arguments
- `plt`: The plot object to which the hyperparameters will be added.
- `hyperparams::Vector{String}`: A vector of strings containing the hyperparameters.

# Returns
- `Nothing`: This function does not return any value.
"""
function plot_hyperparameters(plt, hyperparams::Vector{String})::Nothing
    text = join(hyperparams, "\n")
    plot!(plt, title = "Hyperparameters", xlim = (0, 1), ylim = (0, 1), showaxis = false)
    annotate!(plt, 0.5, 0.5, text, 8) # The last argument sets the font size
    return nothing
end

####################
## Main Plotting ##
####################
function plot_activation_function(plt, kan, p::ComponentArray, stM, i::Int, j::Int, l::Int; xlims::Union{Nothing, Tuple{Real,Real}}=nothing)::Nothing
    #By default plot the grid length of the activation function.
    grid_lims = kan[l].grid_lims
    if isnothing(xlims)
        x = range(grid_lims[1], grid_lims[2], length = 25)
    else
        x = range(xlims[1], xlims[2], length = 25)
    end

    #Get the activation function
    psi=activation_getter(kan, p, stM, l,i,j)

    psi_x=psi.(x)
    #Plot the function with a horizontal line at y=0 for clarity
    if (maximum(psi_x)-minimum(psi_x)<=1e-3)
        mu = sum(psi_x)/length(psi_x)
        plot!(plt, x, psi_x, ylim=(mu -1e-2,mu+1e-2), label = "\\phi_{$l, $i, $j}", lw = 2, xtickfontsize=8)
    else
        plot!(plt, x, psi_x, label = "\\phi_{$l, $i, $j}", lw = 2, xtickfontsize=8) 
    end
    hline!(plt, [0], label = "", color = :black, lw = 1, linestyle = :dash)
    return nothing
end
"""
    plot_KAN_diagram(kan, p::ComponentArray, stM, Xn::Matrix{AbstractFloat})


#Arguments
- 'Xn': This is the input data of the KAN. It will determine xlims of the activation function plot. Note this may not in general be the same as the whole statespace of the UDE or training set. Expects each row to contain the data for that variable.
"""
function plot_KAN_diagram(kan, p::ComponentArray, stM, Xn::Matrix{<:AbstractFloat})
    ranges = activation_range_getter(kan, p, stM, Xn)
    num_layers = length(kan)
    layout = @layout [grid(1, kan[i].in_dims * kan[i].out_dims) for i in 1:num_layers]
    plt = plot(layout = layout, size = (1600, num_layers*500), titlefontsize = 12)

    plot_index = 1
    for l in 1:num_layers
        for i in 1:kan[l].in_dims
            xlims = ranges[l][i]
            for j in 1:kan[l].out_dims
                plot_activation_function(plt[plot_index], kan, p, stM, i, j, l; xlims=xlims)
                plot_index += 1
            end
        end
    end
    return plt
end
"""
    save_training_frame(static_data, UDE_sol::Matrix{Float64}, nn, pM, iter::Int, loss::Vector{Real}, test_loss::Vector{Real}, training_dir::String; save = false)::Plots.Plot

Saves a training frame plot consisting of interaction function surfaces, loss, phase plane, and time series comparison.

# Arguments
- `static_data`: Static data used for plotting interaction surfaces. Can be of type `StaticData_2D` or `StaticData_1D`.
- `UDE_sol::Matrix{Float64}`: Solution matrix from the UDE model. Assumed to be a nx3 matrix where time is in the first column,
                             and predictions for the first and second species are in the second and third columns respectively.
- `nn`: Neural network function used for predictions.
- `pM`: Parameters for the neural network.
- `iter::Int`: Current iteration number.
- `loss::Vector{Real}`: Vector containing the training loss values.
- `test_loss::Vector{Real}`: Vector containing the test loss values. Pass empty vector if not available.
- `training_dir::String`: Directory where the training frames should be saved.
- `save`: Boolean flag indicating whether to save the plot. Default is `false`.

# Returns
- `Plots.Plot`: The generated plot.

# Notes
- The function creates a 2x2 grid plot with the following subplots:
  - Top left: Interaction function surfaces.
  - Top right: Loss plot.
  - Bottom left: Phase plane plot.
  - Bottom right: Time series comparison plot.
- If `save` is `true`, the plot is saved in the specified `training_dir` with the filename format `frame_XXXXX.png`, where `XXXXX` is the zero-padded iteration number.
"""
function plot_training_frame(static_data::Union{StaticData_1D, StaticData_2D}, UDE_sol::Matrix, nn, pM, stM, iter::Int, loss::Vector{Real}, test_loss::Vector{Real}, hyperparams::Vector{String})::Plots.Plot
    #= Its assumed that UDEsol is a nx3 matrix where time is in the first column.
    The second and third columns are the predictions for the first and second species respectively.
    This should be changed in the future to account for any number of species/variables.
    =#
    @suppress begin #This is to stop complaints from the plotting backend
        # KAN prediction

        l = @layout [[a b c];
                     [d{0.3w} e{0.7w}]]
        plt = plot(layout = l, size = (1600, 1200), titlefontsize = 12)



        # Top right: loss
        plot_loss(plt[1], loss, test_loss, iter)

        if isa(static_data, StaticData_2D)
            nn_h = [nn([i, j], pM, stM)[1][] for (i, j) in static_data.xy]
            plot_interaction_surface_2d(plt[2], static_data, nn_h, iter)
        elseif isa(static_data, StaticData_1D)
            nn_h = [nn([i], pM, stM)[1][] for i in static_data.x]
            plot_interaction_surface_1d(plt[2], static_data, nn_h, iter)
        end
        
        plot_hyperparameters(plt[3], hyperparams::Vector{String})

        # Bottom left: Phase plane
        plot_phase_plane(plt[4], UDE_sol, static_data)
    
        # Bottom right: Time series comparison
        plot_time_series(plt[5], UDE_sol, static_data)

        #=
        if save
            # Save figure with iteration number
            savefig(plt, joinpath(training_dir, "frame_$(lpad(iter, 5, '0')).png"))
        end
        =#
        return plt
    end
end
