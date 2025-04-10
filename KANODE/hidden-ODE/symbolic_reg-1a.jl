import SymbolicRegression: SRRegressor,node_to_symbolic
import SymbolicUtils
import MLJ: machine, fit!, predict, report
using Symbolics 
include("hiddenode-1d-1a.jl")
include("hiddenode-1d-1b.jl")
#Generate data
rng = Random.default_rng()
Random.seed!(rng, 3)
@info "Generating data..."
t_test, Xn_test, t_train, Xn_train = generate_data()
u0 = Xn_test[:,1]
##Load the MLP
@info "Loading MLP..."
mlp, p_mlp, stM_mlp  = define_MLP(rng)
@load "hidden-ODE/tests/test_9/checkpoints/MLP_iter30000of30000" p stM
p_mlp, stM_mlp = p, stM
##Load the KAN
@info "Loading KAN..."
kan, unused, unused = define_KAN(rng)
#This was trained before, so we can just load the checkpoint 
@load "hidden-ODE/tests/test_5/checkpoints/KAN_iter30000of30000" p stM 
p_KAN, stM_KAN = p, stM
#Just for displaying the activation functions
Xn = reshape(Xn_train[1,:],1,:)
#@info "Displaying the KAN diagram..."
#display(plot_KAN_diagram(kan, p_KAN, stM_KAN, Xn))
phi1 =  activation_getter(kan, p_KAN, stM_KAN, 1, 1, 1)
phi2 =  activation_getter(kan, p_KAN, stM_KAN, 2, 1, 1)
# Just get the x values from the training set and reshape them into a nx1 matrix
X1 =reshape(Xn_train[1,:],:,1)

model = SRRegressor(
    niterations=50,
    binary_operators=[+, -, *],
    unary_operators=[],
)

#SR on the MLP
y_mlp = mlp(X1', p_mlp, stM_mlp)[1]'
mach_mlp = machine(model, X1, y_mlp)
fit!(mach_mlp)
r_mlp = report(mach_mlp)
best_eq_mlp = r_mlp.equations[r_mlp.best_idx]
mlp_symp = simplify(node_to_symbolic(best_eq_mlp); expand=true)
#SR on the KAN 
y_kan = kan(X1', p, stM)[1]'
mach_kan = machine(model, X1, y_kan)
fit!(mach_kan)
r_kan = report(mach_kan)
best_eq_kan = r_kan.equations[r_kan.best_idx]
kan_symp = simplify(node_to_symbolic(best_eq_kan); expand=true)
#SR on the KAN activation functions

X2 = phi1.(X1)
X3 = phi2.(X2)
mach1 = machine(model, X1, X2)
mach2 = machine(model, X2, X3)
fit!(mach1)
fit!(mach2)
r1 = report(mach1)
r2 = report(mach2)
best_eq_r1 = r1.equations[r1.best_idx]
best_eq_r2 = r2.equations[r2.best_idx] 
expr_r1 = node_to_symbolic(best_eq_r1)
expr_r2 = node_to_symbolic(best_eq_r2)
x1 = Symbolics.get_variables(expr_r1)[1]
composed_expr = substitute(expr_r2, x1 => expr_r1)
composed_symp = simplify(composed_expr; expand=true)
println("Best Equation MLP: ", mlp_symp )
println("Best Equation KAN: ", kan_symp )
println("Composed Equation: ", composed_symp)
#= For SR on the whole KAN =# 

#Note to self April 4 5:40pm. I almost have this working
#but when I was training the MLP I lost the checkpoint. 
#This was in test5, so the KAN was trained correctly but 
#not the MLP. I need to train the MLP again, BUT it im running
#into some problem where it takes a long time to train. I think
#this is due to an error somewhere, but maybe Im wrong? This
#is where you need to pick up where I left off.