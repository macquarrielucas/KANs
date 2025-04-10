These files consist of a few different tests using this KAN-ODEs framework

#1a/1b
`hiddenode-1d-1a.jl` and `symbolic_reg-1a.jl` consist of a basic to recover a logistic function in a predator prey model. However, I want to show how KAN-ODEs can outperform a typical neural network. Currently, it seems that KAN-ODEs doesn't offer any benefit. One thing that should be done is to look at the error of the recovered/hidden term compared over time with the KAN and with the NN and see how they compare. 

##Goal 

Using an SR on MLP, KAN, KAN on the nodes, compare the results for the lotka voltera model with logistic growth to recover the hidden term. 

##Organizing 

`hiddenode-1d-1a.jl` contains the code to train the KAN, saving 
* training iterations (parameters and states)
* training loss profiles
* test loss profiles 

`hiddenode-1a-1b.jl` contains the code to train the MLP, saving
* training iterations (parameters and states)
* training loss profiles
* test loss profiles

# 2d-2a

This test is to show just training on SIR model for the presentation video.