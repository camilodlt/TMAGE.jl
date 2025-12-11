# --- Utility Functions for TPG (to be expanded) ---
"""
    find_program_by_id(tpg::TangledProgramGraph, id::Int)
Finds a TPGProgram by its ID.
"""
function find_program_by_id(tpg::TangledProgramGraph, id::Int)::Union{TPGProgram, Nothing}
    return find_program_by_id(tpg, ProgramID(id))
end

"""
    find_program_by_id(tpg::TangledProgramGraph, id::ProgramID)
Finds a TPGProgram by its ID.
"""
function find_program_by_id(tpg::TangledProgramGraph, id::ProgramID)::Union{TPGProgram, Nothing}
    return get(tpg.programs, id, nothing)
end

"""
    find_team_by_id(tpg::TangledProgramGraph, id::Int)
Finds a TPGTeam by its ID.
"""
function find_team_by_id(tpg::TangledProgramGraph, id::Int)::Union{TPGTeam, Nothing}
    return find_team_by_id(tpg, TeamID(id))
end

"""
    find_team_by_id(tpg::TangledProgramGraph, id::TeamID)
Finds a TPGTeam by its ID.
"""
function find_team_by_id(tpg::TangledProgramGraph, id::TeamID)::Union{TPGTeam, Nothing}
    return get(tpg.teams, id, nothing)
end

function reset_program!(tpg_program::TPGProgram)
    if tpg_program.program !== nothing
        UTCGP.reset_program!(tpg_program.program)
    end
    return tpg_program.program = nothing
end

function reset_programs!(tpg_programs::Vector{TPGProgram})
    for program in tpg_programs
        reset_program!(program)
    end
    return
end

function reset_programs!(tpg_team::TPGTeam)
    for program in tpg_team.programs
        reset_program!(program)
    end
    return
end

function reset_programs!(tpg::TangledProgramGraph)
    for program in values(tpg.programs)
        reset_program!(program)
    end
    return
end


"""
    update_team_action!(tpg::TangledProgramGraph, team_id::TeamID, program_id::ProgramID, new_dest_id::Union{TeamID, Nothing})

Updates the mapping for `program_id` within `team_id`.
- If `new_dest_id` is a `TeamID`, creates or updates the link.
- If `new_dest_id` is `nothing`, removes the link (program becomes a leaf in this team).

This function ensures consistency for:
1. `team.action_map`
2. `team.out_edges` and the destination team's `in_edges`.
3. `program.out_edges` (checking if other teams use this program to point to the same place).
"""
function update_team_action!(tpg::TangledProgramGraph, team_id::TeamID, program_id::ProgramID, new_dest_id::Union{TeamID, Nothing})
    team = tpg.teams[team_id]
    old_dest_id = get(team.action_map, program_id, nothing)

    # If no change, return early
    if old_dest_id == new_dest_id
        return
    end

    # 1. Update the Action Map (Source of Truth)
    # We update this first so subsequent checks on values(team.action_map) reflect the new state.
    if new_dest_id === nothing
        delete!(team.action_map, program_id) # just remove the connection
    else
        team.action_map[program_id] = new_dest_id # update the connection
    end

    # 2. Cleanup Old Connection (if it existed)
    if old_dest_id !== nothing
        # A. Team-Level Edges
        # Check if the team still points to old_dest_id via ANY OTHER program
        if !in(old_dest_id, values(team.action_map))
            delete!(team.out_edges, old_dest_id)
            if haskey(tpg.teams, old_dest_id) # maybe the key is no longer present if gc
                delete!(tpg.teams[old_dest_id].in_edges, team_id) # since no other program in the team points to old_dest_id, we can remove the in edge
            else
                @warn "Team with id $old_dest_id no longer in TPG. A changed action map was pointing to it. Maybe the teams is being GC ?"
            end
        end

        # B. Program-Level Edges
        # Check if the program still points to old_dest_id via ANY OTHER team
        _remove_program_out_edge_if_unused!(tpg, program_id, old_dest_id) # if no other team uses program_id to point to old_dest_id. We will remove the out_edge for that prog.
    end

    # 3. Establish New Connection (if adding/changing) # all Set operations
    return if new_dest_id !== nothing
        # A. Team-Level Edges
        push!(team.out_edges, new_dest_id)
        push!(tpg.teams[new_dest_id].in_edges, team_id)

        # B. Program-Level Edges
        push!(tpg.programs[program_id].out_edges, new_dest_id)
    end
end
