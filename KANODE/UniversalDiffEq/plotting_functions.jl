using Printf


#COLORS 
test_loss_color = :tomato
loss_color=:royalblue1
timeseries_x_color = :lightskyblue2
timeseries_y_color= :royalblue1
dir = @__DIR__

#Structure for passing data which doesn't need to be recalculated
struct StaticData
    x::StepRangeLen
    y::StepRangeLen
    xy::Matrix{Tuple{Float64, Float64}}
    true_h::Matrix{Float64}
    tspan_train::Tuple{Float64, Float64}
    u0::Vector{Float64}
    spinning_rate::Real
end

function find_frame_directory()
    # Create directories if needed
    folder_count = 0
    training_dir = joinpath(dir, "tests", "training_frames_$folder_count")
    #Set up the folder if it doesnt already exist
    if !isdir(joinpath(dir, "tests"))
        println("Making test folder")
        mkdir(joinpath(dir, "tests"))
    end
    if !isdir(training_dir)
        println("Making training_frames_$folder_count folder")
        mkdir(training_dir)
    end
    is_empty = isempty(readdir(training_dir))
    #If its already a directory and its not empty, try the next folder
    while isdir(training_dir) && !is_empty 
        folder_count += 1 
        training_dir = joinpath("tests", "training_frames_$folder_count")
    end
    if !is_empty
        print("Making directory ", string(training_dir))
        mkdir(training_dir)
    end
    training_dir
end

function save_training_frame_2d(static_data::StaticData, model::UDE, nn, pM, iter, loss, test_loss, training_dir; save=false)

    x, y, xy, true_h, tspan_train, u0, spinning_rate = static_data.x, static_data.y, static_data.xy, static_data.true_h, static_data.tspan_train, static_data.u0, static_data.spinning_rate

    # KAN prediction
    nn_h = [nn([i,j], pM, stM)[1][] for (i,j) in  static_data.xy]

    # Get predictions for trajectories
    true_sol = model.data
    UDE_sol = UniversalDiffEq.forecast(model,static_data.u0, model.times)
    print(UDE_sol)
    # Create 2x2 grid plot
    plt = plot(layout=(2,2), size=(1600, 1200), titlefontsize=12)
    
    # Top row: Interaction function surfaces
    plot!(plt[1], static_data.x, static_data.y, static_data.true_h, st=:surface, 
          title="True Interaction h(x,y) compared with KAN (Iteration $iter)",
          xlabel="x", ylabel="y", zlabel="h(x,y)",
          alpha = 0.2, c=:blues,
            camera=(iter*static_data.spinning_rate, 30))
    
    plot!(plt[1], static_data.x,  static_data.y, nn_h, st=:surface, c=:blues,
            label="KAN")
 
    # Top right: loss
    plot!(plt[2], loss, color = loss_color, label = "training loss",
      title = string("Loss Profiles| Loss:", @sprintf("%.4e", loss[end]), "|",
                      "Test_Loss:", @sprintf("%.4e", test_loss[end])), xlabel = "iteration",
      yaxis=:log)
    #plot!(plt[2], test_loss, color = test_loss_color , label = "test loss")

    # Bottom left: Phase plane
    plot!(plt[3], model.data[1,:], model.data[2,:], 
          label="True Solution", lw=2, color=:black,
          xlabel="x", ylabel="y",
          xlim=(0,4.5),ylim=(0,4.5),
           title="Phase Plane")
    plot!(plt[3], UDE_sol[:,2], UDE_sol[:,3], 
          label="KAN Prediction", lw=2, color=:royalblue1, linestyle=:dash)
    scatter!(plt[3], [static_data.u0[1]], [static_data.u0[2]], color=:blue, label="Initial Condition")
    
    # Bottom right: Time series comparison
    plot!(plt[4], model.times, model.data[1,:], label="True x",color=:lightskyblue2, lw=2)
    plot!(plt[4], model.times, model.data[2,:], label="True y",color=:royalblue1, lw=2)
    plot!(plt[4], UDE_sol[:,1], UDE_sol[:,2],   label="KAN x", color=:lightskyblue2, linestyle=:dash, lw=2)
    plot!(plt[4], UDE_sol[:,1], UDE_sol[:,3],   label="KAN y", color=:royalblue1, linestyle=:dash, lw=2)
    vspan!(plt[4], [static_data.tspan_train[1], static_data.tspan_train[2]], alpha=0.2, color=:gray, label="Training Region")
    #=plot!(plt[4], 
        xlabel="Time", ylabel="Population",
        ylim=(0,6),
        title="Time Series Comparison", legend=:topright) =#
    if save
        # Save figure with iteration number
        savefig(plt, joinpath(training_dir, "frame_$(lpad(iter, 5, '0')).png"))
    end
    
    return plt
end
#=
function save_training_frame_1d(p, iter, loss, test_loss,training_dir; save=false)
    # Get predictions for trajectories

    kan_sol = predict_test(p)
    #kan_train_sol = predict(p)
    kan_h = [kan1([i],ComponentArray(p,pM_axis),stM)[1][1] for i in x]
    # Create 2x2 grid plot
    plt = plot(layout=(2,2), size=(1600, 1200), titlefontsize=12)
    
    # Top row: Interaction function surfaces
    plot!(plt[1], x, x.*(1 .- x),
            title="True Interaction: h(x)=x(1-x)/10",
            xlabel="x", ylabel="y",
            label="True h(x)",
            color=:black,
            xlim=(-1,sol_max_x+1))
    plot!(plt[1], x, kan_h, 
            title="KAN Learned Interaction (Iteration $iter)",
            label = "KAN(x,θ)",
            color=:red,
            linestyle=:dash)
    vspan!(plt[1],[0,sol_max_x],alpha=0.2, color=:gray, label="State Space")
  # Top right: loss
    plot!(plt[2], loss, color = loss_color, label = "training loss",
            title = string("Loss Profiles| Loss:", @sprintf("%.4e", loss[end]), "|",
                            "Test_Loss:", @sprintf("%.4e", test_loss[end])), xlabel = "iteration",
            yaxis=:log)
    plot!(plt[2], test_loss, color = test_loss_color , label = "test loss")
    # Bottom left: Phase plane
    plot!(plt[3], true_sol[1,:], true_sol[2,:], 
          label="True Solution", lw=2, color=:black,
          xlabel="x", ylabel="y",
          #xlim=(0,4.5),ylim=(0,4.5),
           title="Phase Plane")
    plot!(plt[3], kan_sol[1,:], kan_sol[2,:], 
          label="KAN Prediction", lw=2, color=:red, linestyle=:dash)
    scatter!(plt[3], [u0[1]], [u0[2]], color=:blue, label="Initial Condition")
    
    # Bottom right: Time series comparison
    plot!(plt[4], t_test, true_sol[1,:], label="True x", color=timeseries_x_color, lw=2)
    plot!(plt[4], t_test, true_sol[2,:], label="True y", color=timeseries_y_color, lw=2)
    plot!(plt[4], kan_sol.t, kan_sol[1,:], label="KAN x", color=timeseries_x_color, linestyle=:dash, lw=2)
    plot!(plt[4], kan_sol.t, kan_sol[2,:], label="KAN y", color=timeseries_y_color, linestyle=:dash, lw=2)
    vspan!(plt[4], [tspan_train[1], tspan_train[2]], alpha=0.2, color=:gray, label="Training Region")
    plot!(plt[4], 
        xlabel="Time", ylabel="Population",
        title="Time Series Comparison", legend=:topright)
    if save
        # Save figure with iteration number
        savefig(plt, joinpath(training_dir, "frame_$(lpad(iter, 5, '0')).png"))
    end
    return plt
end
=#