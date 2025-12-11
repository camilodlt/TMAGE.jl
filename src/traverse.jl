##############################################################
# TRAVERSAL                                                  #
##############################################################

"""
    _traverse_tpg_internal(tpg::TangledProgramGraph, starting_team_ids::Set{TeamID})

Internal function to traverse the TPG starting from a given set of `starting_team_ids`.
It identifies all reachable teams and programs and calculates the shortest path length
from any starting team to each reachable team.

Returns:
- `reachable_teams::Set{TeamID}`: All teams reachable from any starting team.
- `reachable_programs::Set{ProgramID}`: All programs that are part of reachable teams.
- `team_path_lengths::Dict{TeamID, Int}`: Shortest path length from any starting team to each team.
"""
function _traverse_tpg_internal(tpg::TangledProgramGraph, starting_team_ids::Set{TeamID})
    reachable_teams = Set{TeamID}()
    reachable_programs = Set{ProgramID}()
    team_path_lengths = Dict{TeamID, Int}()

    queue = Deque{Tuple{TeamID, Int}}()

    # Initialize queue with all provided starting teams
    for start_id in starting_team_ids
        if haskey(tpg.teams, start_id) # Ensure the starting team actually exists
            push!(reachable_teams, start_id)
            team_path_lengths[start_id] = 0
            push!(queue, (start_id, 0))
        else
            @warn "Starting team $start_id not found in TPG.teams. Skipping traversal from this point."
        end
    end

    while !isempty(queue)
        current_team_id, current_depth = popfirst!(queue)

        # Add programs of the current team to reachable programs
        current_team = tpg.teams[current_team_id]
        for program in current_team.programs
            push!(reachable_programs, program.id)
        end

        # Explore outgoing connections
        for next_team_id in current_team.out_edges
            if haskey(tpg.teams, next_team_id) # Ensure the destination team actually exists
                if !(next_team_id in reachable_teams)
                    push!(reachable_teams, next_team_id)
                    team_path_lengths[next_team_id] = current_depth + 1
                    push!(queue, (next_team_id, current_depth + 1))
                else
                    # If already visited, check if this path is shorter
                    if haskey(team_path_lengths, next_team_id) && team_path_lengths[next_team_id] > current_depth + 1
                        team_path_lengths[next_team_id] = current_depth + 1
                        # Re-queueing ensures that if a shorter path is found, it's propagated.
                        push!(queue, (next_team_id, current_depth + 1))
                    end
                end
            else
                @warn "Team $(current_team_id) points to non-existent team $(next_team_id) in its out_edges."
            end
        end
    end

    return reachable_teams, reachable_programs, team_path_lengths
end

"""
    _traverse_tpg_from_roots(tpg::TangledProgramGraph)

Traverses the TPG starting from all known root teams to identify all reachable
teams and programs. It also calculates the shortest path length from any root
to each reachable team.

Returns:
- `reachable_teams::Set{TeamID}`: All teams reachable from any root.
- `reachable_programs::Set{ProgramID}`: All programs that are part of reachable teams.
- `team_path_lengths::Dict{TeamID, Int}`: Shortest path length from any root to each team.
"""
function _traverse_tpg_from_roots(tpg::TangledProgramGraph)
    return _traverse_tpg_internal(tpg, tpg.root_teams)
end

"""
    _traverse_tpg_from_roots(tpg::TangledProgramGraph, root_id::TeamID)

Traverses the TPG starting from a specific `root_id` to identify all reachable
teams and programs from that root. It also calculates the shortest path length
from this root to each reachable team.

Returns:
- `reachable_teams::Set{TeamID}`: All teams reachable from the specified root.
- `reachable_programs::Set{ProgramID}`: All programs that are part of teams reachable from the specified root.
- `team_path_lengths::Dict{TeamID, Int}`: Shortest path length from the specified root to each team.
"""
function _traverse_tpg_from_roots(tpg::TangledProgramGraph, root_id::TeamID)
    return _traverse_tpg_internal(tpg, Set([root_id]))
end
