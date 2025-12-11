using Statistics

##############################################################
# TPG INTEGRITY CHECKS AND STATISTICS                        #
##############################################################

"""
    TPG_Integrity_Report
Holds the results of a TPG integrity check, including verification status,
orphaned components, and graph statistics.
"""
mutable struct TPG_Integrity_Report
    is_consistent::Bool
    total_teams::Int
    total_programs::Int
    reachable_teams_count::Int
    reachable_programs_count::Int
    orphaned_teams::Set{TeamID}
    orphaned_programs::Set{ProgramID}

    # Consistency issues
    team_in_edge_mismatches::Vector{String}
    team_out_edge_mismatches::Vector{String}
    program_in_edge_mismatches::Vector{String}
    program_out_edge_mismatches::Vector{String}
    action_map_program_id_mismatches::Vector{String}
    action_map_team_id_mismatches::Vector{String}
    root_team_existence_mismatches::Vector{String}

    # Statistics
    coverage_teams_percent::Float64
    coverage_programs_percent::Float64
    min_path_length::Union{Int, Nothing}
    max_path_length::Union{Int, Nothing}
    mean_path_length::Union{Float64, Nothing}
    std_path_length::Union{Float64, Nothing}

    function TPG_Integrity_Report()
        return new(
            true, 0, 0, 0, 0, Set{TeamID}(), Set{ProgramID}(),
            Vector{String}(), Vector{String}(), Vector{String}(), Vector{String}(),
            Vector{String}(), Vector{String}(), Vector{String}(),
            0.0, 0.0, nothing, nothing, nothing, nothing
        )
    end
end

function Base.show(io::IO, report::TPG_Integrity_Report)
    println(io, "--- TPG Integrity Report ---")
    println(io, "Consistency: $(report.is_consistent ? "✅ Consistent" : "❌ Inconsistent")")
    println(io, "Total Teams: $(report.total_teams), Total Programs: $(report.total_programs)")
    println(io, "Reachable Teams: $(report.reachable_teams_count) ($(round(report.coverage_teams_percent, digits = 2))% Coverage)")
    println(io, "Reachable Programs: $(report.reachable_programs_count) ($(round(report.coverage_programs_percent, digits = 2))% Coverage)")

    if !report.is_consistent
        println(io, "\n--- Inconsistency Details ---")
        if !isempty(report.orphaned_teams)
            println(io, "  Orphaned Teams ($(length(report.orphaned_teams))): $(report.orphaned_teams)")
        end
        if !isempty(report.orphaned_programs)
            println(io, "  Orphaned Programs ($(length(report.orphaned_programs))): $(report.orphaned_programs)")
        end
        if !isempty(report.root_team_existence_mismatches)
            println(io, "  Root Team Existence Mismatches:")
            for msg in report.root_team_existence_mismatches
                println(io, "    - $(msg)")
            end
        end
        if !isempty(report.team_in_edge_mismatches)
            println(io, "  Team In-Edge Mismatches ($(length(report.team_in_edge_mismatches))):")
            for msg in report.team_in_edge_mismatches
                println(io, "    - $(msg)")
            end
        end
        if !isempty(report.team_out_edge_mismatches)
            println(io, "  Team Out-Edge Mismatches ($(length(report.team_out_edge_mismatches))):")
            for msg in report.team_out_edge_mismatches
                println(io, "    - $(msg)")
            end
        end
        if !isempty(report.program_in_edge_mismatches)
            println(io, "  Program In-Edge Mismatches ($(length(report.program_in_edge_mismatches))):")
            for msg in report.program_in_edge_mismatches
                println(io, "    - $(msg)")
            end
        end
        if !isempty(report.program_out_edge_mismatches)
            println(io, "  Program Out-Edge Mismatches ($(length(report.program_out_edge_mismatches))):")
            for msg in report.program_out_edge_mismatches
                println(io, "    - $(msg)")
            end
        end
        if !isempty(report.action_map_program_id_mismatches)
            println(io, "  Action Map Program ID Mismatches ($(length(report.action_map_program_id_mismatches))):")
            for msg in report.action_map_program_id_mismatches
                println(io, "    - $(msg)")
            end
        end
        if !isempty(report.action_map_team_id_mismatches)
            println(io, "  Action Map Team ID Mismatches ($(length(report.action_map_team_id_mismatches))):")
            for msg in report.action_map_team_id_mismatches
                println(io, "    - $(msg)")
            end
        end
    end

    println(io, "\n--- Graph Statistics ---")
    return if report.min_path_length !== nothing
        println(io, "Shortest Path Length: $(report.min_path_length)")
        println(io, "Longest Path Length: $(report.max_path_length)")
        println(io, "Mean Path Length: $(round(report.mean_path_length, digits = 2)) ± $(round(report.std_path_length, digits = 2))")
    else
        println(io, "No path length statistics available (graph might be empty or unreachable).")
    end
end


"""
    verify_tpg_integrity!(tpg::TangledProgramGraph; cleanup_orphans::Bool = false, cache::Union{Nothing, TPGEvaluationCache} = nothing)

Performs a comprehensive integrity check on the Tangled Program Graph.
It traverses the graph from all root teams, verifies connections, identifies
orphaned components, and can optionally perform garbage collection.

Arguments:
- `tpg::TangledProgramGraph`: The TPG to verify.
- `cleanup_orphans::Bool`: If true, identified orphaned teams and programs will be removed.
- `cache::Union{Nothing, TPGEvaluationCache}`: Optional evaluation cache, used if cleaning programs.

Returns:
- `report::TPG_Integrity_Report`: A detailed report of the TPG's integrity.
"""
function verify_tpg_integrity!(tpg::TangledProgramGraph; cleanup_orphans::Bool = false, cache::Union{Nothing, TPGEvaluationCache} = nothing)
    report = TPG_Integrity_Report()
    report.total_teams = length(tpg.teams)
    report.total_programs = length(tpg.programs)

    # 1. Traverse the graph from all root nodes
    reachable_teams_set, reachable_programs_set, team_path_lengths = _traverse_tpg_from_roots(tpg)

    report.reachable_teams_count = length(reachable_teams_set)
    report.reachable_programs_count = length(reachable_programs_set)

    # Calculate coverage
    if report.total_teams > 0
        report.coverage_teams_percent = (report.reachable_teams_count / report.total_teams) * 100
    end
    if report.total_programs > 0
        report.coverage_programs_percent = (report.reachable_programs_count / report.total_programs) * 100
    end

    # Calculate path length statistics
    if !isempty(team_path_lengths)
        path_lengths = collect(values(team_path_lengths))
        report.min_path_length = minimum(path_lengths)
        report.max_path_length = maximum(path_lengths)
        report.mean_path_length = mean(path_lengths)
        report.std_path_length = std(path_lengths)
    end

    # 2. Identify orphaned components
    report.orphaned_teams = setdiff(keys(tpg.teams), reachable_teams_set)
    report.orphaned_programs = setdiff(keys(tpg.programs), reachable_programs_set)

    if !isempty(report.orphaned_teams) || !isempty(report.orphaned_programs)
        report.is_consistent = false
    end

    # 3. Perform integrity checks on reachable components
    # Check root team existence
    for root_id in tpg.root_teams
        if !haskey(tpg.teams, root_id)
            push!(report.root_team_existence_mismatches, "Root team $(root_id) defined but does not exist in tpg.teams.")
            report.is_consistent = false
        end
    end

    for team_id in reachable_teams_set
        team = tpg.teams[team_id]

        # Verify action_map program IDs exist in team.programs
        team_program_ids = Set([p.id for p in team.programs])
        for pid in keys(team.action_map)
            if !in(pid, team_program_ids)
                push!(report.action_map_program_id_mismatches, "Team $(team_id) action_map references program $(pid) not in team.programs.")
                report.is_consistent = false
            end
        end

        # Verify team.out_edges
        expected_out_edges = Set{TeamID}(values(team.action_map))
        if team.out_edges != expected_out_edges
            push!(report.team_out_edge_mismatches, "Team $(team_id) out_edges $(team.out_edges) does not match expected $(expected_out_edges) from action_map.")
            report.is_consistent = false
        end

        # Verify team.in_edges (against programs that point to it)
        # Check all other teams in the TPG that should point to this team
        expected_in_edges = Set{TeamID}()
        for other_tid in keys(tpg.teams)
            if other_tid != team_id
                other_team = tpg.teams[other_tid]
                for dest_tid in values(other_team.action_map)
                    if dest_tid == team_id
                        push!(expected_in_edges, other_tid)
                    end
                end
            end
        end
        if team.in_edges != expected_in_edges
            push!(report.team_in_edge_mismatches, "Team $(team_id) in_edges $(team.in_edges) does not match expected $(expected_in_edges) from other teams' action_maps.")
            report.is_consistent = false
        end
    end

    for program_id in reachable_programs_set
        program = tpg.programs[program_id]

        # Verify program.in_edges
        expected_in_edges = Set{TeamID}()
        for team_id in keys(tpg.teams)
            team = tpg.teams[team_id]
            if in(program_id, [p.id for p in team.programs])
                push!(expected_in_edges, team_id)
            end
        end
        if program.in_edges != expected_in_edges
            push!(report.program_in_edge_mismatches, "Program $(program_id) in_edges $(program.in_edges) does not match expected $(expected_in_edges) from teams containing it.")
            report.is_consistent = false
        end

        # Verify program.out_edges
        expected_out_edges = Set{TeamID}()
        for team_id in keys(tpg.teams)
            team = tpg.teams[team_id]
            if get(team.action_map, program_id, nothing) !== nothing
                push!(expected_out_edges, team.action_map[program_id])
            end
        end
        if program.out_edges != expected_out_edges
            push!(report.program_out_edge_mismatches, "Program $(program_id) out_edges $(program.out_edges) does not match expected $(expected_out_edges) from action_maps.")
            report.is_consistent = false
        end
    end

    # cache info
    if cache.mode == LRUCacheMode
        hits = 0
        misses = 0
        for (k, lru_cache) in cache.program_caches
            i = cache_info(lru_cache)
            hits += i.hits
            misses += i.misses
        end
        println("CACHE HITS : $hits")
        println("CACHE MISSES : $misses")
    end

    # 4. Optional Cleanup
    if cleanup_orphans && (!isempty(report.orphaned_teams) || !isempty(report.orphaned_programs))
        @info "Cleaning up orphaned teams and programs."
        _gc_tpg!(tpg; cache = cache)

        # After cleanup, re-verify to ensure full consistency
        @info "Re-verifying TPG after cleanup."
        re_report = verify_tpg_integrity!(tpg; cleanup_orphans = false, cache = cache) # No further cleanup in re-run
        return re_report # Return the new report after cleanup
    end

    return report
end
