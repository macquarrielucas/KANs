import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report

include("hiddeode-1d.jl")

#Random
rng = Random.default_rng()
Random.seed!(rng, 3)
println("Generating data...")
t_test, Xn_test, t_train, Xn_train = generate_data()
u0 = Xn_test[:,1]
println("Initializing KAN...")
kan, pM, stM, layer_width, grid_size = define_KAN(rng)
pM_data     = getdata(ComponentArray(pM))
pM_axis     = getaxes(ComponentArray(pM))
p = ComponentArray(pM_data, pM_axis) 

@load "Trained_model_25000" p stM 
Xn = reshape(Xn_train[1,:],1,:)
display(plot_KAN_diagram(kan, p, stM, Xn))
ranges = activation_range_getter(kan,p,stM, Xn)
phi =  activation_getter(kan, p, stM, 1, 1,3)::Function
# Dataset with two named features:
X =reshape(Xn_train[1,:],:,1)

# and one target:
y = @. phi(X)


model = SRRegressor(
    niterations=50,
    binary_operators=[+, -, *,/],
    unary_operators=[exp],
)

mach = machine(model, X, y)

fit!(mach)

report(mach)
