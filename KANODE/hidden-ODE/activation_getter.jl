"""
    This function takes in the parameters, state, and model for
    a KAN and returns the activation function at the specified
    layer. We say that the coordinates of the activation functions
    are given by i,j,l where i in the input node, j is the output 
    node, and l is the layer.

"""
function activation_getter(kan, pM, stM, i,j,l)::Function
    layer = kan[l]
    p = pM[l]
    st = stM[l]
    grid_len = layer.grid_len
    grid_lims = layer.grid_lims
    
    function activation_function(x<:Real)
        #TODO This is not correct. This is a start, but it  returns a whole
        #matrix when the idea is to return a single value. Remember that 
        #we just want the activation function.
        size_in  = size(x)                          # [..., I]
        size_out = (layer.out_dims, size_in[2:end]...,) # [..., O]

        x = reshape(x, layer.in_dims, :)
        K = size(x, 2)

        x_norm = layer.normalizer.(x)
        x_resh = reshape(x_norm, 1, :)
        basis  = layer.basis_func(x_resh, st.grid, layer.denominator)
        basis  = reshape(basis, layer.grid_len * layer.in_dims, K)
        spline = p.C * basis
    end
    return activation_function
end