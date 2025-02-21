
# PREDICTION FUNCTION
function single_shooting_predict(model::Function,p, u0::Vector{Float32},times::Vector{Float32})
    tspan = (times[1],times[end])
    prob = ODEProblem(model, u0, tspan, p, saveat=times)
    sol = solve(prob, Tsit5(), verbose = false);
    X = Array(sol)
    return X
end

"""
    This function is used to make a prediction using the multiple shooting method.
    
    This outputs the same number of datapoints as the input data, but the first 
    pred length is replaced by the final predicted value. That is 
    
            x^i_f -> x^(i+1)_o

    This is for implementing with the loss function. In this implementation, there
    is no need to consider constraints on the final prediction point and initial 
    observed values, since we choose our subintervals based on what timepoints we have
    data for.

    pred_length should be greater than one. but consider if setting pred_length=1 should
    be the same as single shooting.
"""
function multiple_shooting_predict(model::Function,p,pred_length::Int, times::Vector{Float32}, data::Matrix{Float32})

    #Solve the ODE on the subinterval \tau_j to \tau_{j+1}
    function predict_mini(model::Function, u,tsteps,parameters)
        tspan =  (tsteps[1],tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        prob=ODEProblem(model, u, tspan, parameters, saveat=tsteps)
        sol = OrdinaryDiffEq.solve(prob, abstol=1e-6, reltol=1e-6)
        X = Array(sol)
        return X
    end 

    #Calculate the subinternals
    inds = 1:pred_length
    tspan = 1:pred_length
    Xn = data[:,1]
    #By reversing the loop we can avoid overwriting the final
    #point of each subinterval. 
    for tau in 1:pred_length:(size(times)[1])
        #The initial point of the interval
        u0 = data[:,tau]
        #If the interval is the first one
        #=if tau ==1 
            inds = tau:(tau+pred_length)
            tspan = times[inds]
            #The data matrix for the interval
            u1hat = predict_mini(model,u0,tspan,p)
            Xn = hcat(Xn,u1hat)
        =#
        #If the interval is in the bounds of times
        if (size(times)[1]-tau) >= pred_length
            inds = tau:(tau+pred_length)
            tspan = times[inds]
            #The data matrix for the interval
            u1hat = predict_mini(model,u0,tspan,p)[:,2:end] #throw away the first value
            Xn = hcat(Xn,u1hat)
        #If the interval would be greater than the size of times
        else
            inds = tau:size(times)[1]
            if length(inds) == 1
                break
            else
                tspan = times[inds]
                #The data matrix for the interval
                u1hat = predict_mini(model,u0,tspan,p)[:,2:end]
                Xn = hcat(Xn,u1hat)
            end
        end
    end

    return Xn
end
"""
    multiple_shooting_loss(p, pred_length::Int, times::Vector{<:AbstractFloat})::Real

Calculate the multiple shooting loss for the given parameters. (this can be rewritten to use an external multiple_shooting_predict function)

# Arguments
- `UDE`: The universal differential equation model. Should be a function
- `p`: The parameters of the model.
- `pred_length::Int`: The length of the prediction interval.
- `times::Vector{<:AbstractFloat}`: The time points at which the solution is evaluated.

# Returns
- `Real`: The calculated loss value.
"""
function multiple_shooting_loss(model::Function, p, pred_length::Int, times::Vector{<:AbstractFloat}, data::Matrix{Float32})::Real
    mean(abs2, data.- multiple_shooting_predict(model, p, 
                                            pred_length, #pred_length
                                            times,
                                            data) )
end
function single_shooting_loss(model::Function, p, times::Vector{Float32}, data::Matrix{Float32})::Real
    mean(abs2, data.- single_shooting_predict(model, p, 
                                            data[:,1], #Initial conditions
                                            times) )
end
# LOSS FUNCTIONS
function reg_loss(p, act_reg=1.0, entropy_reg=0.0)::Real
    l1_temp=(abs.(p))
    activation_loss=sum(l1_temp)
    #This entropy was not mentioned in the paper i believe,
    #so i assuming its some optional thing they played with.
    entropy_temp=l1_temp/activation_loss
    entropy_loss=-sum(entropy_temp.*log.(entropy_temp))
    total_reg_loss=activation_loss*act_reg+entropy_loss*entropy_reg
    return total_reg_loss
end



#overall loss
"""
    loss_train(model::Function, p, sparse_on::Int)::Real

    Calculate the training loss for the given parameters.
    
    # Arguments
    - `model::Function`: The universal differential equation model.
    - `p`: The parameters of the model.
    - `sparse_on`: This is a flag. Takes 1 or 0. 1 uses regularization. (Should be changes in the future)
"""
function loss_train(model::Function, p,times::Vector{<:AbstractFloat}, data::Matrix{Float32}; sparse_on::Int=0)::Real
    #loss_temp=single_shooting_loss(p)
    #loss_temp=multiple_shooting_loss(model, p, 5, times,data)
    loss_temp=single_shooting_loss(model, p, times,data)
    if sparse_on==1
        loss_temp+=reg_loss(p, 5e-4, 0) #if we have sparsity enabled, add the reg loss
    end
    return loss_temp
end
#=
function loss_test(model::Function, p, data::Matrix{Float32})::Real
    mean(abs2, data .- Array(single_shooting_predict(model,p)))
end
=#