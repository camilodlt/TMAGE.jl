##############################################################
# EVALUATION                                                 #
##############################################################

# --- Evaluation ---

"""
    evaluate(program::TPGProgram, input::Any, cache::TPGEvaluationCache, shared_inputs::SharedInput, meta_library::MetaLibrary, model_architecture::modelArchitecture)
Evaluates a single TPGProgram, leveraging the global cache and shared inputs.
The underlying UTCGP.Program is decoded lazily on first evaluation.
"""
function evaluate(tpg_program::TPGProgram, input::Any, cache::TPGEvaluationCache, shared_inputs::SharedInput, meta_library::MetaLibrary, model_architecture::modelArchitecture)
    # If cache is disabled, or mode is NoCache, directly evaluate
    if !is_cache_enabled(cache) || cache.mode == NoCache
        if tpg_program.program === nothing
            tpg_program.program = UTCGP.decode_with_output_nodes(tpg_program.genome, meta_library, model_architecture, shared_inputs).programs[1]
        end
        UTCGP.replace_shared_inputs!(tpg_program.program.program_inputs, input)
        result = UTCGP.evaluate_program(tpg_program.program, model_architecture.chromosomes_types, meta_library)
        reset_program!(tpg_program)
        return result
    end

    program_input = input # Simplification for now
    input_hash = hash(program_input) # Hash the input for caching
    cached_value = get_cached_value(cache, tpg_program.id, input_hash)

    if cached_value !== nothing
        return cached_value
    else
        if tpg_program.program === nothing
            tpg_program.program = UTCGP.decode_with_output_nodes(tpg_program.genome, meta_library, model_architecture, shared_inputs).programs[1]
        end
        UTCGP.replace_shared_inputs!(tpg_program.program.program_inputs, input)
        result = UTCGP.evaluate_program(tpg_program.program, model_architecture.chromosomes_types, meta_library)
        set_cached_value!(cache, tpg_program.id, input_hash, result)
        reset_program!(tpg_program)
        return result
    end
end

"""
    evaluate(tpg_team::TPGTeam, input::Any, cache::TPGEvaluationCache, shared_inputs::SharedInput, meta_library::MetaLibrary, model_architecture::modelArchitecture)
Evaluates a TPGTeam. Each program in the team evaluates the input, bids, and the winning program's
action determines the next step. The output of the winning program is the team's output.
"""
function evaluate(tpg_team::TPGTeam, input::Any, cache::TPGEvaluationCache, shared_inputs::SharedInput, meta_library::MetaLibrary, model_architecture::modelArchitecture)
    bids = Dict{ProgramID, Any}()
    for program in tpg_team.programs
        bids[program.id] = evaluate(program, input, cache, shared_inputs, meta_library, model_architecture)
    end

    winning_program_id = ProgramID(-1)
    max_bid_value = -Inf

    for (program_id, result) in bids
        if typeof(result) <: Number
            if result > max_bid_value
                max_bid_value = result
                winning_program_id = program_id
            end
        else
            if winning_program_id.val == -1
                winning_program_id = program_id
                max_bid_value = NaN
                break
            end
        end
    end

    if winning_program_id.val == -1
        error("TPGTeam $(tpg_team.id) could not determine a winning program for input. All programs returned non-numeric or uncomparable results.")
    end

    team_output = bids[winning_program_id]

    next_team_id = get(tpg_team.action_map, winning_program_id, nothing)

    return team_output, next_team_id, winning_program_id # New: return winning_program_id
end

"""
    evaluate(tpg::TangledProgramGraph, root_team_id::TeamID, input::Any, shared_inputs::SharedInput, meta_library::MetaLibrary, model_architecture::modelArchitecture; cache_mode::CacheMode = PerInputCache, lru_max_size::Int = 1000)
Evaluates a TangledProgramGraph starting from a given root team, using the provided shared inputs.
This function simulates the execution path through the TPG, utilizing caching based on `cache_mode`.
Includes loop detection. If a loop is detected, the current winning program's action (if any) or its output is returned.
"""
function evaluate(
        tpg::TangledProgramGraph, root_team_id::TeamID, input::Any, shared_inputs::SharedInput, meta_library::MetaLibrary, model_architecture::modelArchitecture;
        cache::TPGEvaluationCache
    )
    if !haskey(tpg.teams, root_team_id)
        error("Root team with ID $root_team_id not found in TPG.")
    end

    current_team_id = root_team_id

    results_path = []
    final_output = nothing # To store the last team's output
    final_program_action = nothing # To store the last winning program's action

    # Keep track of visited teams in this evaluation path to detect loops
    visited_teams_in_path = Set{TeamID}()

    while current_team_id !== nothing
        # Loop detection
        if current_team_id in visited_teams_in_path
            @info "Loop detected! Returning final result as if current team were a leaf. Team ID: $(current_team_id)"
            # If a loop is detected, return the last known program action or program output
            if final_program_action !== nothing
                return final_program_action, results_path
            else
                return final_output, results_path
            end
        end
        push!(visited_teams_in_path, current_team_id)

        current_team = tpg.teams[current_team_id]
        team_output, next_team_id, winning_program_id = evaluate(current_team, input, cache, shared_inputs, meta_library, model_architecture)
        push!(results_path, (current_team.id.val, team_output)) # Store Int in path for external compatibility/readability
        final_output = team_output # The output of the current team

        # Determine the winning program for this team to get its action
        # Note: evaluate(current_team...) already determined the winner to get outputs, but didn't return the object.
        # We can fetch the program object using the ID returned.
        winning_program_obj = tpg.programs[winning_program_id]

        if winning_program_obj !== nothing
            final_program_action = winning_program_obj.action # Store the winning program's action
        else
            final_program_action = nothing # No winning program means no action
        end


        if next_team_id !== nothing && haskey(tpg.teams, next_team_id)
            current_team_id = next_team_id
        else
            # This is a leaf team (or an invalid next_team_id)
            # Return the winning program's action if it has one, otherwise its output
            if final_program_action !== nothing
                return final_program_action, results_path
            else
                return final_output, results_path
            end
            current_team_id = nothing # Ensure loop terminates
        end
    end

    # If the loop finishes naturally (no next_team_id), return the last collected values
    if final_program_action !== nothing
        return final_program_action, results_path
    else
        return final_output, results_path
    end
end

function evaluate(tpg::TangledProgramGraph, root_team_id::Int, input::Any, shared_inputs::SharedInput, meta_library::MetaLibrary, model_architecture::modelArchitecture; cache_mode::CacheMode = PerInputCache, lru_max_size::Int = 1000)
    return evaluate(tpg, TeamID(root_team_id), input, shared_inputs, meta_library, model_architecture, cache_mode = cache_mode, lru_max_size = lru_max_size)
end
