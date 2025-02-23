
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