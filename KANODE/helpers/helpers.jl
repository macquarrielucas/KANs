

"""
    find_frame_directory()::String

Creates a new directory for storing training frames, checkpoints. The function ensures that the 
    directory structure exists and creates new directories as needed. It starts 
    with "training_frames_0" and increments the folder count until it finds an 
    empty directory.

# Returns
- `String`: The path to the newly created or existing empty directory.
"""
function find_tests_directory(dir::String)::String 
    # Create directories if needed
    folder_count = 0
    training_dir = joinpath(dir, "tests", "test_$folder_count")
    # Set up the folder if it doesn't already exist
    if !isdir(joinpath(dir, "tests"))
        println("Making test folder")
        mkdir(joinpath(dir, "tests"))
    end
    if !isdir(training_dir)
        println("Making"*training_dir*"folder")
        mkdir(training_dir)
    end
    is_empty = isempty(readdir(training_dir))
    # If it's already a directory and it's not empty, try the next folder
    while isdir(training_dir) && !is_empty 
        folder_count += 1 
        training_dir = joinpath(dir, "tests", "test_$folder_count")
    end
    if !is_empty
        println("Making directory ", string(training_dir))
        mkdir(training_dir)
    end
    training_dir
end

function get_training_dir(dir::String)
    training_dir = find_tests_directory(dir)
    println("Saving frames to: ", training_dir)
    return training_dir
end 

function save_model_parameters(i, N_iter, p::ComponentArray, stM, dir::String)::Nothing 
    if !isdir(joinpath(dir, "checkpoints"))
        mkdir(joinpath(dir, "checkpoints"))
    end
    filename = joinpath(dir, "checkpoints", "iter$i"*"of$N_iter")
    @save filename p stM 
end
"""
    This function takes in the parameters, state, and model for
    a KAN and returns the activation function at the specified
    layer. We say that the coordinates of the activation functions
    are given by i,j,l where i is the input node, j is the output 
    node, and l is the layer.

"""
function activation_getter(kan, p::ComponentArray, stM, l::Int, i::Int, j::Int)::Function
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

    layer_name = keys(p)[l]
    p = p[layer_name]

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
"""
    activation_range_getter(kan, p::ComponentArray, stM, Xn::Matrix{<:AbstractFloat})::Vector{Vector{Tuple}}

Compute the range of activation values for each layer in a KAN model.

# Arguments
- `kan`: A vector representing the layers of the neural network. Each element corresponds to a layer and should have an `in_dims` field indicating the number of input dimensions.
- `p::ComponentArray`: A `ComponentArray` containing the parameters for each layer of the network. The keys of `p` should match the layer names.
- `stM`: A vector representing the state of each layer in the network. Each element corresponds to the state of a layer.
- `Xn::Matrix{<:AbstractFloat}`: A matrix of input data where each column represents a sample and each row represents a feature.

# Returns
- `Vector{Vector{Tuple}}`: A vector of vectors, where each inner vector contains tuples representing the minimum and maximum activation values for each input dimension of the corresponding layer.

# Description
This function calculates the range of activation values for each layer in a neural network by passing the input data through the network. For each layer, it computes the minimum and maximum values of the activations for each input dimension. The first layer uses the input data directly, while subsequent layers use the outputs of the previous layers as inputs.

# Example
```julia
kan = [...]  # Define your neural network layers
p = ComponentArray(...)  # Define your parameters
stM = [...]  # Define your layer states
Xn = rand(10, 100)  # Generate some input data

ranges = activation_range_getter(kan, p, stM, Xn)


"""
function activation_range_getter(kan, p::ComponentArray, stM, Xn::Matrix{<:AbstractFloat})::Vector{Vector{Tuple}}
    # Check if the number of input dimensions in Xn matches kan[1].in_dims
    if size(Xn, 1) != kan[1].in_dims
        error("Dimension mismatch: Xn has $(size(Xn, 1)) inputs, but the first layer expects $(kan[1].in_dims) inputs.")
    end

    num_layers = length(kan)
    ranges = Vector{Vector{Tuple}}()
    #We need to pass the input data through each layer to determine the range of each
    #activator function.
    for l in 1:num_layers
        init_ranges_vector = Vector{Tuple}()
        if l != 1 
            #Pass the data through the previous layer to find 
            layer = kan[l-1]
            st_layer = stM[l-1]
            layer_name = keys(p)[l-1]
            p_layer = p[layer_name]
            #Get the values at the output nodes
            inputs = layer(Xn, p_layer, st_layer)[1]
            Xn=inputs
        else 
            inputs = Xn #The first layer is just the input data
        end 
        for i in 1:kan[l].in_dims
            xlims = (minimum(inputs[i,:]), maximum(inputs[i,:])) 
            push!(init_ranges_vector, xlims)
        end

        push!(ranges, init_ranges_vector)
    end
    return ranges
end

