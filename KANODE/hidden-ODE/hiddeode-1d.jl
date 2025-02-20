# PACKAGES AND INCLUSIONS
using ChainRulesCore
using ComponentArrays
using ComponentArrays: getaxes
using ComponentArrays: getdata
using ConcreteStructs
using DiffEqFlux
using Flux
using Flux: mae, mean, update!
using ForwardDiff
using LinearAlgebra
using MAT
using ModelingToolkit
using NNlib
using Optim
using Optimisers
using OrdinaryDiffEq
using Plots
using Printf
using ProgressBars
using Random
using WeightInitializers
using Zygote
using Lux
pythonplot()

include("../helpers/plotting_functions.jl")
include("loss_functions.jl")
# DIRECTORY
dir         = @__DIR__
dir         = "$dir/"
cd(dir)
add_path    = "test/"
#=
mkpath(join(dir,add_path,"figs"))
mkpath(join(dir,add_path,"checkpoints"))
mkpath(join(dir,add_path,"training_frames"))
=#
# KAN PACKAGE LOAD
include("../Lotka-Volterra/src/KolmogorovArnold.jl")
using .KolmogorovArnold

#Random
rng = Random.default_rng()
Random.seed!(rng, 3)

function h(x)
    #The hidden function that we are trying to learn
    x*(1-x)
end

function lotka!(du,u,p,t) 
    du[1] = h(u[1])- 0.5*u[1]*u[2]
    du[2] = 0.5*u[1]*u[2] - 0.03*u[2]
end

#data generation parameters
dt=0.1
tspan_test = (0.0, 500)
tspan_train=(0.0, 100)
u0 = [1.0f0, 1.0f0]
p_=[]
prob = ODEProblem(lotka!, u0,tspan_test,p_)

#generate training data, split into train/test
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = dt, verbose = false)
# Calculate the index to split the solution into training and test sets based on the training time span
end_index = Int64(floor(length(solution.t) * tspan_train[2] / tspan_test[2])) + 1
t_test = solution.t #full dataset
t_train=t_test[1:end_index] #training cut
X = Array(solution)
Xn = deepcopy(X) 
plot(solution)
# Define KAN
basis_func = rbf      # rbf, rswaf
normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign
layer_width=5
grid_size=10
#This KAN looks like
# psi(phi1(x) + phi2(x) + phi3(x) + ... + phi10(x))
kan1 = Lux.Chain(
    KDense( 1, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  1, grid_size; use_base_act = true, basis_func, normalizer),
)
pM , stM  = Lux.setup(rng, kan1)
pM_data     = getdata(ComponentArray(pM))
pM_axis     = getaxes(ComponentArray(pM))
p = ComponentArray(pM_data, pM_axis) 
#p = (deepcopy(pM_data))./1e5 ;

# CONSTRUCT KAN-ODES
function kanode!(du, u, p, t)
    kan1_(x) = kan1([x], p, stM)[1][1]
    du[1] = kan1_(u[1]) - 0.5*u[1]*u[2]
    du[2] = 0.5*u[1]*u[2] - 0.03*u[2]
end

# PREDICTION FUNCTION
function predict(p)
    prob = ODEProblem(kanode!, u0, tspan_train,p, saveat=dt)
    sol = solve(prob, Tsit5(), verbose = false);
end
#Prediction function over the test set.
function predict_test(p)
    prob = ODEProblem(kanode!, u0, tspan_test,p, saveat=dt)
    sol = solve(prob, Tsit5(), verbose = false);
end

function multiple_shooting_loss(p, pred_length::Int, times::Vector{<:AbstractFloat})::Real

    #Solve the ODE on the subinterval \tau_j to \tau_{j+1}
    function predict_mini(u,tsteps,parameters)
        tspan =  (tsteps[1],tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        prob=ODEProblem(kanode!, u, tspan, parameters, saveat=tsteps)
        sol = OrdinaryDiffEq.solve(prob, abstol=1e-6, reltol=1e-6)
        X = Array(sol)
        return X
    end 

    #Calculate the subinternals then the loss
    loss = 0 
    inds = 1:pred_length
    tspan = 1:pred_length
    for tau in 1:pred_length:(size(times)[1])
        #If the interval is in the bounds of times
        if ( size(times)[1]-tau) >= pred_length
            inds = tau:(tau+pred_length)
            tspan = times[inds]
        #If the interval would be greater than the size of times
        else
            inds = tau:size(times)[1]
            tspan = times[inds]
        end 
        #The tau'th point of the data matrix
        u0 = Xn[:,tau]
        #The data matrix for the interval
        u1 = Xn[:,inds]
        u1hat = predict_mini(u0,tspan,p)
        for i in 1:length(inds) 
            loss += sum((u1[:,i].-u1hat[:,i]).^2)/length(Xn)
        end 
    end
    return loss
end

# LOSS FUNCTIONS
function reg_loss(p, act_reg=1.0, entropy_reg=1.0)::Real
    l1_temp=(abs.(p))
    activation_loss=sum(l1_temp)
    #This entropy was not mentioned in the paper i believe,
    #so i assuming its some optional thing they played with.
    entropy_temp=l1_temp/activation_loss
    entropy_loss=-sum(entropy_temp.*log.(entropy_temp))
    total_reg_loss=activation_loss*act_reg+entropy_loss*entropy_reg
    return total_reg_loss
end


function single_shooting_loss(p)::Real
    mean(abs2, Xn[:, 1:end_index].- predict(p)) 
end
#overall loss
sparse_on = 0
function loss_train(p)::Real
    #loss_temp=single_shooting_loss(p)
    loss_temp=multiple_shooting_loss(p, 5, t_train)
    if sparse_on==1
        loss_temp+=reg_loss(p, 5e-4, 0) #if we have sparsity enabled, add the reg loss
    end
    return loss_temp
end
#=
function loss_train(p)
    mean(abs2, Xn[:, 1:end_index] .- Array(predict(p)))
end
=#
function loss_test(p)::Real
    mean(abs2, Xn .- Array(predict_test(p)))
end


##Plotting
sol_max_x =maximum([x[1] for x in solution.u]) 
#Bounds on the interaction plot
x = range(-1, sol_max_x+1, length=40)
# True interaction function   
true_h = [h(i) for i in x]

observation_data = [solution.t [u[1] for u in solution.u] [u[2] for u in solution.u]]
static_data = StaticData_1D(observation_data, 
                            x,
                            true_h,
                            tspan_train, 
                            u0, 
                            sol_max_x)



SAVE_ON = false
if SAVE_ON
    training_dir = find_frame_directory()
else
    training_dir=""
end
#opt = Flux.Momentum(1e-3, 0.9)
opt = Flux.Adam(1e-4)
N_iter = 10000
iterator = ProgressBar(1:N_iter)

#Stuff to track loss and test loss
l = Real[]
l_test=Real[]
p_list = []
@time for i in iterator
    # GRADIENT COMPUTATION
    grad = Zygote.gradient(loss_train, p)[1]

    # UPDATE WITH ADAM OPTIMIZER
    update!(opt, p, grad)


    # CALLBACK
    loss_curr=deepcopy(loss_train(p))
    loss_curr_test=deepcopy(loss_test(p))
    set_description(iterator, string(
        "Iter:", i, 
        "| Loss:", @sprintf("%.2f", loss_curr), 
        "| Test_Loss:", @sprintf("%.2f", loss_curr_test), 
        "|"
    ))

    append!(l, loss_curr)
    append!(l_test, loss_curr_test)
    #append!(p_list, [deepcopy(p)])


    if i%10 == 0 || i==1
        #Turn the data into an nx3 matrix for the plotting function
        UDE_forecast = predict_test(p)
        UDE_sol=[UDE_forecast.t [u[1] for u in UDE_forecast.u] [u[2] for u in UDE_forecast.u]]
        print(training_dir)
        @time display(save_training_frame(static_data, UDE_sol, kan1,p,i, l,l_test,training_dir; save=SAVE_ON))
    end
    # SAVE
    #callback(i)
end