##############################################################
# MUTATIONS                                                  #
##############################################################

# --- Config ---
"""
    TPGMutationConfig
Configuration for various mutation probabilities in a Tangled Program Graph.
"""
mutable struct TPGMutationConfig
    program_mutation_rate::Float64 # Probability of mutating a node within an underlying UTCGP program
    add_program_to_team_rate::Float64 # Probability of adding a new program to a team
    remove_program_from_team_rate::Float64 # Probability of removing a program from a team
    mutate_action_map_rate::Float64 # Probability of changing an action map entry
    mutate_program_action_rate::Float64 # New: Probability of changing a program's assigned action

    function TPGMutationConfig(
            program_mutation_rate::Float64 = 0.1,
            add_program_to_team_rate::Float64 = 0.1,
            remove_program_from_team_rate::Float64 = 0.1,
            mutate_action_map_rate::Float64 = 0.1,
            mutate_program_action_rate::Float64 = 0.1
        )
        return new(
            program_mutation_rate,
            add_program_to_team_rate,
            remove_program_from_team_rate,
            mutate_action_map_rate,
            mutate_program_action_rate,
        )
    end
end


# --- Mutation Strategies ---

abstract type AbstractMutationStrategy end

"""
    TPGMutationStrategy
Original TPG-like strategy. Copies a team to create a new root.
Mutations (add/remove program, change action map) happen on this new team.
Existing programs are shared unless modified.
"""
struct TPGMutationStrategy <: AbstractMutationStrategy end

"""
    GraphMutationStrategy
Default strategy. Copies a path from the root to a target node deep in the graph.
This allows mutating a program or team deep in the hierarchy while preserving the original graph
by creating a new branch (new root -> ... -> new target).
"""
struct GraphMutationStrategy <: AbstractMutationStrategy end


# ---Mutate a MAGE Program ---

"""
    mutate_program_in_tpg!(tpg::TangledProgramGraph, tpg_program_id::ProgramID, model_architecture::modelArchitecture, meta_library::MetaLibrary, node_config::nodeConfig, shared_inputs::SharedInput)

Classical GP mutation, we mutate the genome in a program.

Mutates the underlying `UTGenome` of a `TPGProgram` within the TPG.
Resets the `program` field to `nothing` to force re-decoding on next evaluation.
Resets the program.

Unsafe because it mutates inplace so call with a new program.
Errors if the new program is not in the TPG
"""
function unsafe_mutate_program_in_tpg!(tpg::TangledProgramGraph, tpg_program_id::ProgramID, model_architecture::modelArchitecture, meta_library::MetaLibrary, node_config::nodeConfig, shared_inputs::SharedInput)
    tpg_program = find_program_by_id(tpg, tpg_program_id)
    if tpg_program === nothing
        @warn "TPGProgram with ID $tpg_program_id not found for mutation."
        return error("Graph is broken ?")
    end

    # Mutate the underlying genome
    UTCGP.numbered_mutation!(tpg_program.genome, UTCGP.NumberedMutationArgs(1), model_architecture, meta_library, shared_inputs)
    UTCGP.reset_genome!(tpg_program.genome)

    # Invalidate the decoded program, so it's re-decoded on next evaluation
    tpg_program.program = nothing
    @debug "Program mutated pid : $(tpg_program.id)"
    return true
end

# --- Random Additions to a TPG ---
# Add a random program to a team.
# Add a random team to a TPG

"""
    add_random_program_to_team!(tpg::TangledProgramGraph, team_id::TeamID, model_architecture::modelArchitecture, meta_library::MetaLibrary, node_config::nodeConfig, shared_inputs::SharedInput)

Creates a new random `UTGenome`, wraps it in a `TPGProgram`, and adds it to a specified team.

The new program's action will be randomly selected from `tpg.actions` or `nothing`.

Returns the id of the new program
"""
function add_random_program_to_team!(tpg::TangledProgramGraph, team_id::TeamID, model_architecture::modelArchitecture, meta_library::MetaLibrary, node_config::nodeConfig, shared_inputs::SharedInput)
    team = tpg.teams[team_id]

    # Create a new random genome
    _, new_genome = UTCGP.make_evolvable_utgenome(model_architecture, meta_library, node_config)
    UTCGP.initialize_genome!(new_genome)
    UTCGP.correct_all_nodes!(new_genome, model_architecture, meta_library, shared_inputs)
    UTCGP.fix_all_output_nodes!(new_genome)

    # Add the new program to the TPG and the team
    new_tpg_program = add_program!(tpg, new_genome)
    push!(team.programs, new_tpg_program)
    push!(new_tpg_program.in_edges, team.id) # Update edge to new program
    @debug "Added new program $(new_tpg_program.id) to team $(team.id) with action $(new_tpg_program.action)."
    return new_tpg_program.id
end


"""
    add_nonrandom_program_to_team!(tpg::TangledProgramGraph, team_id::TeamID, model_architecture::modelArchitecture, meta_library::MetaLibrary, node_config::nodeConfig, shared_inputs::SharedInput)

Adds an existing program to a team. Programs are filtered to only allow for new programs (not in the team) to be added.

Action map is not changed, so by design the added program does not point to a team.
"""
function add_nonrandom_program_to_team!(tpg::TangledProgramGraph, team_id::TeamID, model_architecture::modelArchitecture, meta_library::MetaLibrary, node_config::nodeConfig, shared_inputs::SharedInput)
    team = find_team_by_id(tpg, team_id)
    team_progs = map(p -> p.id, team.programs)
    possible_pids = filter(p -> !(p in team_progs), collect(keys(tpg.programs)))
    if !isempty(possible_pids)
        # Add the new program to the TPG and the team
        choice = rand(possible_pids)
        program = find_program_by_id(tpg, choice)
        _add_program_to_team!(tpg, team, program, nothing)
        @debug "Added new program $(program.id) to team $(team.id) . Present action : $(program.action)."
        return choice
    else
        @warn "Tried to add existing program to team but couldn't because there are no possible program choices"
        return nothing
    end
end

"""
    add_random_team_to_tpg!(tpg::TangledProgramGraph, model_architecture::modelArchitecture, meta_library::MetaLibrary, node_config::nodeConfig, shared_inputs::SharedInput)

Creates a new `TPGTeam` with a few random programs and potentially connects it to an existing team.

Returns the id of the new team
"""
function add_random_team_to_tpg!(
        tpg::TangledProgramGraph, model_architecture::modelArchitecture, meta_library::MetaLibrary, node_config::nodeConfig, shared_inputs::SharedInput;
        up_to_n_new_progs = 3,
        prob_out_con = 0.2
    ) # don't know if this is still needed ?
    @assert up_to_n_new_progs >= 1
    # Create some programs for the new team
    new_program_ids = ProgramID[]
    num_initial_programs = rand(1:up_to_n_new_progs) # 1 to 3 programs per new team

    for _ in 1:num_initial_programs
        _, new_genome = UTCGP.make_evolvable_utgenome(model_architecture, meta_library, node_config)
        UTCGP.initialize_genome!(new_genome)
        UTCGP.correct_all_nodes!(new_genome, model_architecture, meta_library, shared_inputs)
        UTCGP.fix_all_output_nodes!(new_genome)
        new_tpg_program = add_program!(tpg, new_genome) # assigns action if applicable
        push!(new_program_ids, new_tpg_program.id)
    end

    # Create the new team
    new_team = add_team!(tpg, new_program_ids) # this updates the edges in programs

    # Optionally connect the new team to a existing team
    if !isempty(tpg.teams) && length(tpg.teams) > 1 && rand() < prob_out_con
        existing_team_id = rand(collect(keys(tpg.teams)))
        if existing_team_id != new_team.id
            prog = rand(new_team.programs)
            update_team_action!(tpg, new_team.id, prog.id, existing_team_id)
            return new_team.id
        end
    end

    @debug "Added new team $(new_team.id) action_map : $(new_team.action_map)."
    return new_team.id
end

# --- Remove from TPG (programs and teams) ---

"""
    remove_program_from_team!(tpg::TangledProgramGraph, team_id::TeamID, program_id::ProgramID)

Removes a specific `TPGProgram` from a `TPGTeam`.

If the program is part of the team's action map,
that entry is also removed.

Updates :
- Program's in edge from team
- Program in list of program
- Team's action map involving that program
- If there was a mapping :
    - removes the teams out edge if applicable (no other prog in team points to dst)
    - dst in edge if applicable (no other prog in team points to dst)
    - program's out edge if applicable (no other team uses prog to point to dst)
"""
function remove_program_from_team!(tpg::TangledProgramGraph, team_id::TeamID, program_id::ProgramID)
    team = find_team_by_id(tpg, team_id)
    program = find_program_by_id(tpg, program_id)

    # Remove program from team's program list
    filter!(p -> p.id != program_id, team.programs)

    # Remove from action map if present, update all edges
    update_team_action!(tpg, team_id, program_id, nothing)

    # Remove the in edge for the prog
    delete!(program.in_edges, team_id)

    @debug "Removed program $(program_id) from team $(team_id)."
    return true
end

"""
    remove_team_from_tpg!(tpg::TangledProgramGraph, team_id::TeamID; force::Bool = false)

Removes a team from the TPG and updates all relevant connections.
If `force` is true, the team is removed even if it still has incoming edges.
This should be used with caution, primarily by internal garbage collection processes.
"""
function remove_team_from_tpg!(tpg::TangledProgramGraph, team_id::TeamID; force::Bool = false)
    team_to_remove = find_team_by_id(tpg, team_id)
    if team_to_remove === nothing
        @warn "Attempted to remove non-existent team $(team_id). Skipping."
        return
    end

    if !force && !isempty(team_to_remove.in_edges)
        @warn "Attempting to remove team $(team_id) which is still referenced by other teams $(team_to_remove.in_edges). Skipping."
        return
    end

    # 1. Update outgoing connections: remove this team from in_edges of teams it points to
    for next_tid in team_to_remove.out_edges
        if haskey(tpg.teams, next_tid)
            delete!(tpg.teams[next_tid].in_edges, team_id)
        end
    end

    # 2. Update programs: remove this team from in_edges of programs it contains
    ids_of_programs_to_remove = map(x -> x.id, team_to_remove.programs) # we have to collect if before since we will mutate the team.programs in place
    for program_id in ids_of_programs_to_remove
        remove_program_from_team!(tpg, team_to_remove.id, program_id) # empties the action map. Updates the connections
    end

    # 3. Remove from root_teams if it was a root
    delete!(tpg.root_teams, team_id)

    # 4. Finally, delete the team itself
    delete!(tpg.teams, team_id)
    @debug "Removed team with id $(team_id)"
    return
end


# --- TPGMutationStrategy Logic ---
function _mutate_single_offspring!(tpg::TangledProgramGraph, parent_root_id::TeamID, strategy::TPGMutationStrategy, config::TPGMutationConfig, ma::modelArchitecture, ml::MetaLibrary, nc::nodeConfig, si::SharedInput, cache::TPGEvaluationCache)
    # 1. Clone the parent root team (Shallow Copy)
    new_root = copy_team(tpg, parent_root_id)
    set_root_team!(tpg, new_root.id)

    # 2. Mutate the NEW root team
    if rand() < config.remove_program_from_team_rate && length(new_root.programs) > 1 # remove a program
        program_to_remove = rand(new_root.programs)
        remove_program_from_team!(tpg, new_root.id, program_to_remove.id)
    end # by doing this first, we avoid making a new prog, changing it's action map and after that it gets deleted ...

    if rand() < config.add_program_to_team_rate # add a new program
        # add_random_program_to_team!(tpg, new_root.id, ma, ml, nc, si)
        add_nonrandom_program_to_team!(tpg, new_root.id, ma, ml, nc, si)
    end # by doing this first, it's action or action map can be changed by the rest of the variations below

    # since we mutate the new_root.programs in place
    # we iterate over a fixed list
    # else the for loop has bad behavior
    programs_in_new_root_before_mutations = map(x -> x.id, new_root.programs)
    for program_id in programs_in_new_root_before_mutations
        if rand() < config.program_mutation_rate # mutate directly a program
            # Create a mutated copy
            new_prog = copy_program(tpg, program_id) # empty edges
            unsafe_mutate_program_in_tpg!(tpg, new_prog.id, ma, ml, nc, si)
            # Replace in team: remove old, add new
            _replace_program_in_team!(tpg, new_root.id, program_id, new_prog.id)
        end
        if rand() < config.mutate_program_action_rate # program unchanged but the action associated changes
            new_prog = copy_program(tpg, program_id)
            copy_cache!(cache, program_id, new_prog.id) # the new program is the same but has a diff action, we can recover all the cache from the last
            unsafe_mutate_program_action!(tpg, new_prog.id)
            _replace_program_in_team!(tpg, new_root.id, program_id, new_prog.id)
        end
    end

    if rand() < config.mutate_action_map_rate # change the mapping for the new_root
        unsafe_mutate_action_map!(tpg, new_root.id)
    end
    return new_root.id
end

# --- GraphMutationStrategy Logic ---
# function _mutate_single_offspring!(tpg::TangledProgramGraph, parent_root_id::TeamID, strategy::GraphMutationStrategy, config::TPGMutationConfig, ma::modelArchitecture, ml::MetaLibrary, nc::nodeConfig, si::SharedInput)
#     # 1. Identify reachable components
#     reachable_teams, reachable_programs = get_reachable_components(tpg, parent_root_id)

#     # 2. Select a target for mutation
#     # We can mutate a team (action map, structure) or a program (genome).
#     # Let's flip a coin or use rates. For simplicity, we pick one target type based on config rates sum?
#     # Or just pick any component uniformly.

#     all_targets = []
#     for tid in reachable_teams
#         push!(all_targets, (tid, :team))
#     end
#     for pid in reachable_programs
#         push!(all_targets, (pid, :program))
#     end

#     if isempty(all_targets)
#         return
#     end

#     target_id, target_type = rand(all_targets)

#     # 3. Copy Path from root to target
#     path_teams = get_path_to_component(tpg, parent_root_id, target_id)
#     if isempty(path_teams)
#         return
#     end # Should not happen if reachable

#     # "Zipper": Copy teams along the path and re-link them.
#     # previous_copy_id will track the ID of the newly created copy of the current step's parent
#     # so we can link it to the next step's copy.

#     # The new root will be the copy of the first team in path
#     new_root_id = nothing
#     previous_copy_id = nothing

#     # Map to track original_id -> copy_id for teams in the path
#     path_copies = Dict{TeamID, TeamID}()

#     for i in 1:length(path_teams)
#         original_tid = path_teams[i]

#         # Create shallow copy of the team
#         new_team = copy_team(tpg, original_tid)
#         path_copies[original_tid] = new_team.id

#         if i == 1
#             new_root_id = new_team.id
#             set_root_team!(tpg, new_root_id)
#         else
#             # Link previous copy to this new copy
#             # We need to find which program in 'previous_copy' pointed to 'original_tid'
#             # and update it to point to 'new_team.id'
#             prev_original_tid = path_teams[i - 1]
#             prev_copy_tid = path_copies[prev_original_tid]
#             prev_copy_team = tpg.teams[prev_copy_tid]

#             # Find programs that pointed to the next step in the original path
#             # and update them in the copy to point to the new copy
#             for (prog_id, dest_id) in prev_copy_team.action_map
#                 if dest_id == original_tid
#                     prev_copy_team.action_map[prog_id] = new_team.id

#                     # Update edges
#                     # Remove edge to old destination (original_tid) from copy
#                     delete!(tpg.teams[original_tid].in_edges, prev_copy_tid) # Was added by copy_team!
#                     delete!(prev_copy_team.out_edges, original_tid)
#                     _remove_program_out_edge_if_unused!(tpg, prog_id, original_tid, ignore_team = prev_copy_tid)

#                     # Add edge to new destination
#                     push!(tpg.teams[new_team.id].in_edges, prev_copy_tid)
#                     push!(prev_copy_team.out_edges, new_team.id)
#                     push!(tpg.programs[prog_id].out_edges, new_team.id)
#                 end
#             end
#         end
#     end

#     # 4. Mutate the target (which is now part of the new path)
#     # The target is at the end of the chain.
#     last_copy_id = path_copies[path_teams[end]]

#     return if target_type == :team
#         # The target team is 'last_copy_id'
#         # Apply team mutations to this copy
#         if rand() < config.add_program_to_team_rate
#             add_random_program_to_team!(tpg, last_copy_id, ma, ml, nc, si)
#         end
#         if rand() < config.remove_program_from_team_rate
#             # Logic to ensure >= 1 program is handled inside remove_program_from_team? No, check here.
#             team = tpg.teams[last_copy_id]
#             if length(team.programs) > 1
#                 p = rand(team.programs)
#                 remove_program_from_team!(tpg, last_copy_id, p.id)
#             end
#         end
#         if rand() < config.mutate_action_map_rate
#             mutate_action_map!(tpg, last_copy_id)
#         end

#     elseif target_type == :program
#         # The target program is IN the team 'last_copy_id'
#         # We must deep copy this program and replace it in 'last_copy_id'
#         # target_id is the ProgramID

#         # Deep copy and mutate
#         new_prog = copy_program(tpg, target_id)

#         if rand() < config.program_node_mutation_rate
#             unsafe_mutate_program_in_tpg!(tpg, new_prog.id, ma, ml, nc, si)
#         end
#         if rand() < config.mutate_program_action_rate
#             mutate_program_action!(tpg, new_prog.id)
#         end

#         # Replace in the team
#         _replace_program_in_team!(tpg, last_copy_id, target_id, new_prog.id)
#     end
# end


# MUTATION LOW LEVEL

# Helper to swap a program in a team (used for mutations that require isolation)
function _add_program_to_team!(tpg::TangledProgramGraph, team::TPGTeam, program::TPGProgram, mapped_to::Union{Nothing, TeamID})
    ids = [p.id for p in team.programs]
    return if !(program.id in ids)
        push!(team.programs, program)
        push!(program.in_edges, team.id)
        update_team_action!(tpg, team.id, program.id, mapped_to)
    else
        @error "Program $(program.id) was already in team $(team.id)"
    end

end
function _replace_program_in_team!(tpg::TangledProgramGraph, team_id::TeamID, old_pid::ProgramID, new_pid::ProgramID)
    team = find_team_by_id(tpg, team_id)
    new_prog = find_program_by_id(tpg, new_pid)
    old_map = get(team.action_map, old_pid, nothing)
    remove_program_from_team!(tpg, team_id, old_pid) # remove old program and connections
    _add_program_to_team!(tpg, team, new_prog, old_map) # add new program and put back connection if there was
    return new_prog.id
end

"""
    mutate_action_map!(tpg::TangledProgramGraph, team_id::TeamID)

Modifies the action map of a specified team, either by adding a new mapping,
changing an existing one, or removing one.
"""
function unsafe_mutate_action_map!(tpg::TangledProgramGraph, team_id::TeamID)
    team = find_team_by_id(tpg, team_id)
    possible_actions = ["add", "change", "remove"]
    action = rand(possible_actions)

    if action == "add" && !isempty(tpg.teams) && length(team.action_map) < length(team.programs) # Can only add if space available
        program_ids_without_action = [p.id for p in team.programs if !haskey(team.action_map, p.id)]
        if isempty(program_ids_without_action)
            @debug "No programs available to add to action map for team $(team_id)."
            return false
        end
        program_id_to_map = rand(program_ids_without_action)
        next_team_id = rand(filter(t -> t != team.id, collect(keys(tpg.teams)))) # Connect to a random existing team but not itself
        update_team_action!(tpg, team_id, program_id_to_map, next_team_id)
        @debug "Action map: Added mapping for program $(program_id_to_map) to team $(next_team_id) in team $(team_id)."
        return true
    elseif action == "change" && !isempty(team.action_map) && !isempty(tpg.teams)
        program_id_to_change = rand(collect(keys(team.action_map)))
        old_next_team_id = team.action_map[program_id_to_change]
        possible_next_teams = filter(id -> id != old_next_team_id && id != team_id, collect(keys(tpg.teams))) # avoid making a loop
        new_next_team_id = rand(possible_next_teams)
        if isempty(possible_next_teams)
            @debug "No alternative teams to change action map to for team $(team_id)."
            return false
        end
        update_team_action!(tpg, team_id, program_id_to_change, new_next_team_id)
        @debug "Action map: Changed mapping for program $(program_id_to_change) from $(old_next_team_id) to $(new_next_team_id) in team $(team_id)."
        return true
    elseif action == "remove" && !isempty(team.action_map)
        program_id_to_remove = rand(collect(keys(team.action_map)))
        update_team_action!(tpg, team_id, program_id_to_remove, nothing)
        @debug "Action map: Removed mapping for program $(program_id_to_remove) in team $(team_id)."
        return true
    end
    @debug "Action map mutation did nothing for team $(team_id)."
    return false
end

"""
    mutate_program_action!(tpg::TangledProgramGraph, program_id::ProgramID)
Changes the action assigned to a `TPGProgram` to a random available action from `tpg.actions`,
or `nothing` if `tpg.actions` is empty.
"""
function unsafe_mutate_program_action!(tpg::TangledProgramGraph, program_id::ProgramID)
    tpg_program = find_program_by_id(tpg, program_id)
    if tpg_program === nothing
        @warn "TPGProgram with ID $program_id not found for action mutation."
        return false
    end

    if isempty(tpg.actions)
        tpg_program.action = nothing
        @debug "Program $(program_id): Action set to nothing (no actions available in TPG)."
    else
        old_action = tpg_program.action
        new_action = rand(tpg.actions)
        if new_action == old_action && length(tpg.actions) > 1
            # Try to pick a different action if possible
            possible_new_actions = filter(a -> a != old_action, tpg.actions)
            if !isempty(possible_new_actions)
                new_action = rand(possible_new_actions)
            end
        end
        tpg_program.action = new_action
        @debug "Program $(program_id): Action changed from $(old_action) to $(new_action)."
    end
    return true
end
