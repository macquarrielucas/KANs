
function get_training_dir(SAVE_ON::Bool)
    if SAVE_ON
        dir = @__DIR__
        training_dir = find_frame_directory(dir)
        println("Saving frames to: ", training_dir)
    else
        training_dir=""
    end
    return training_dir
end 

"""
    This function takes in the parameters, state, and model for
    a KAN and returns the activation function at the specified
    layer. We say that the coordinates of the activation functions
    are given by i,j,l where i is the input node, j is the output 
    node, and l is the layer.

"""
function activation_getter(kan, pM::ComponentArray, stM, i::Int, j::Int, l::Int)::Function
    # Check if l is within bounds
    if l < 1 || l > length(kan)
        throw(ArgumentError("Layer index l=$l is out of bounds. It should be 1 <= l <= $(length(kan))."))
    end

    layer = kan[l]
    st = stM[l]

    # Check if i and j are within bounds
    if i < 1 || i > layer.in_dims
        throw(ArgumentError("Input node index i=$i is out of bounds. It should be 1 <= i <= $(layer.in_dims)."))
    end
    if j < 1 || j > layer.out_dims
        throw(ArgumentError("Output node index j=$j is out of bounds. It should be 1 <= j <= $(layer.out_dims)."))
    end

    layer_name = keys(pM)[l]
    p = pM[layer_name]

    C = Array(p.C)
    W = Array(p.W)
    grid_len = layer.grid_len
    grid_lims = layer.grid_lims

    function activation_function(x::Real; use_norm::Bool = true)::Real       
        if use_norm
            x_norm = layer.normalizer(x)
            basis = layer.basis_func(x_norm, st.grid, layer.denominator)
        else      
            basis = layer.basis_func(x, st.grid, layer.denominator)
        end 

        # Submatrix for the given input i
        submatrix = C[:, (i-1)*grid_len+1:i*grid_len]
        # Row for the given output j 
        coeffecients = submatrix[j, :]
        # Sum a_{i,j} * phi_{i,j}(x,z)
        spline = coeffecients' * basis

        # Is this layer using weighted activator? (use_base_act == true?)
        if typeof(layer).parameters[1]
            weight = W[j, i] * layer.base_act(x)
            return spline + weight 
        else
            return spline
        end
    end

    return activation_function
end