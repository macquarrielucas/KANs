struct StaticData
    x::StepRangeLen
    y::StepRangeLen
    xy::Matrix{Tuple{Float64, Float64}}
    true_h::Matrix{Float64}
    true_sol::Any  # Replace `Any` with the actual type of `solution`
    tspan_train::Tuple{Float64, Float64}
    u0::Vector{Float64}
end
#COLORS 
test_loss_color = :tomato
loss_color=:royalblue1
timeseries_x_color = :lightskyblue2
timeseries_y_color= :royalblue1
spinning_rate = 0.5
function save_training_frame_2d(static_data::StaticData, model::UDE, p, iter, loss, test_loss, training_dir; save=false)

    x, y, xy, true_h, true_sol, tspan_train, u0 = static_data.x, static_data.y, static_data.xy, static_data.true_h, static_data.true_sol, static_data.tspan_train, static_data.u0

    # KAN prediction
    kan_h = [kan1([i,j], ComponentArray(p,pM_axis), stM)[1][] for (i,j) in xy]

    # Get predictions for trajectories
    true_sol = solution 
    kan_sol = predict_test(p)
    #kan_train_sol = predict(p)
    
    # Create 2x2 grid plot
    plt = plot(layout=(2,2), size=(1600, 1200), titlefontsize=12)
    
    # Top row: Interaction function surfaces
    plot!(plt[1], x, y, true_h, st=:surface, 
          title="True Interaction h(x,y) compared with KAN (Iteration $iter)",
          xlabel="x", ylabel="y", zlabel="h(x,y)",
          alpha = 0.2, c=:blues,
            camera=(iter*spinning_rate, 30))
    
    plot!(plt[1], x, y, kan_h, st=:surface, c=:blues,
            label="KAN") 
      # Top right: loss
    plot!(plt[2], loss, color = loss_color, label = "training loss",
        title = "Loss Profiles", xlabel = "iteration",
        yaxis=:log)
    plot!(plt[2], test_loss, color = test_loss_color , label = "test loss")
    # Bottom left: Phase plane
    plot!(plt[3], true_sol[1,:], true_sol[2,:], 
          label="True Solution", lw=2, color=:black,
          xlabel="x", ylabel="y",
          xlim=(0,4.5),ylim=(0,4.5),
           title="Phase Plane")
    plot!(plt[3], kan_sol[1,:], kan_sol[2,:], 
          label="KAN Prediction", lw=2, color=:royalblue1, linestyle=:dash)
    scatter!(plt[3], [u0[1]], [u0[2]], color=:blue, label="Initial Condition")
    
    # Bottom right: Time series comparison
    plot!(plt[4], true_sol.t, true_sol[1,:], label="True x", color=:lightskyblue2, lw=2)
    plot!(plt[4], true_sol.t, true_sol[2,:], label="True y", color=:royalblue1, lw=2)
    plot!(plt[4], kan_sol.t, kan_sol[1,:], label="KAN x", color=:lightskyblue2, linestyle=:dash, lw=2)
    plot!(plt[4], kan_sol.t, kan_sol[2,:], label="KAN y", color=:royalblue1, linestyle=:dash, lw=2)
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