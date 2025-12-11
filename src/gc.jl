##############################################################
# GC                                                         #
##############################################################

"""
    _gc_tpg!(tpg::TangledProgramGraph; cache::Union{Nothing, TPGEvaluationCache} = nothing)

Performs garbage collection on the TPG. It iteratively identifies and removes
all teams and programs that are unreachable from any root team. This function
ensures that all necessary updates to `in_edges` and `out_edges` are made
and uses the standard removal interfaces (`remove_team_from_tpg!`,
`remove_program_from_team!`).

Arguments:
- `tpg::TangledProgramGraph`: The TPG to clean.
- `cache::Union{Nothing, TPGEvaluationCache}`: Optional evaluation cache, used to clear program caches.

Returns:
- `teams_cleaned::Int`: Number of teams removed.
- `programs_cleaned::Int`: Number of programs removed.
"""
function _gc_tpg!(tpg::TangledProgramGraph; cache::Union{Nothing, TPGEvaluationCache} = nothing)
    teams_cleaned_count = 0
    programs_cleaned_count = 0

    # Iterate until no more teams or programs can be removed
    it = 1
    while true
        # 1. Traverse to find current reachable components
        reachable_teams_set, reachable_programs_set, _ = _traverse_tpg_from_roots(tpg)

        # 2. Identify currently orphaned teams and programs
        current_orphaned_teams = setdiff(keys(tpg.teams), reachable_teams_set)

        # A program is truly orphaned if it's not reachable AND no reachable team contains it.
        # This is already captured by reachable_programs_set.
        current_orphaned_programs = setdiff(keys(tpg.programs), reachable_programs_set)

        if isempty(current_orphaned_teams) && isempty(current_orphaned_programs)
            @info "GC done : pass $it. Teams $(teams_cleaned_count). Programs $(programs_cleaned_count)"
            break # No more orphans to clean, terminate loop
        end

        # 3. Remove orphaned teams
        teams_removed_in_this_pass = 0
        for tid in current_orphaned_teams
            if haskey(tpg.teams, tid) # Double check it still exists before trying to remove
                remove_team_from_tpg!(tpg, tid; force = true) # Use force=true for GC
                teams_removed_in_this_pass += 1
            end
        end
        teams_cleaned_count += teams_removed_in_this_pass

        # 4. Remove orphaned programs
        programs_removed_in_this_pass = 0
        for pid in current_orphaned_programs
            if haskey(tpg.programs, pid) && isempty(tpg.programs[pid].in_edges) # Programs must have no incoming team references
                delete!(tpg.programs, pid)
                if cache !== nothing
                    delete!(cache.program_caches, pid)
                end
                programs_removed_in_this_pass += 1
            end
        end
        programs_cleaned_count += programs_removed_in_this_pass

        # If nothing was removed in this pass, but orphans still exist,
        # it means there's a circular dependency of orphans or an unhandled case.
        # This should ideally not happen if the traversal correctly identifies reachability.
        if teams_removed_in_this_pass == 0 && programs_removed_in_this_pass == 0
            @warn "GC could not remove all orphaned components in this pass. Remaining orphaned teams: $(current_orphaned_teams), programs: $(current_orphaned_programs)."
            break
        end

        @info "GC : pass $(it) Teams removed : $(teams_removed_in_this_pass). Programs removed $(programs_removed_in_this_pass)"
        it += 1
    end

    return teams_cleaned_count, programs_cleaned_count
end
