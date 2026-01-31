module TMAGE

using UTCGP
using DataStructures
using StatsBase
using LRUCache
using GraphViz
using TimerOutputs

import UTCGP:
    AbstractProgram,
    Program,
    evaluate,
    reset_program!,
    reset_programs!,
    MetaLibrary,
    UTGenome,
    make_evolvable_utgenome,
    initialize_genome!,
    correct_all_nodes!,
    fix_all_output_nodes!,
    numbered_mutation!,
    modelArchitecture,
    nodeConfig,
    SharedInput

export TangledProgramGraph,
    TPGProgram,
    TPGTeam,
    ProgramID,
    TeamID,
    evaluate,
    reset_program!,
    reset_programs!,
    set_root_team!,
    TPGEvaluationCache,
    init_cache,
    enable_cache!,
    disable_cache!,
    is_cache_enabled,
    clear_cache!,
    get_cached_value,
    set_cached_value!,
    find_program_by_id,
    find_team_by_id,
    TPGMutationConfig,
    AbstractMutationStrategy,
    TPGMutationStrategy,
    GraphMutationStrategy,
    mutate_tpg!,
    mutate_program_in_tpg!,
    add_random_program_to_team!,
    remove_program_from_team!,
    add_random_team_to_tpg!,
    remove_team_from_tpg!,
    mutate_action_map!,
    mutate_program_action!,
    update_actions!,
    update_team_action!,
    CacheMode,
    NoCache, PerInputCache, LRUCacheMode,
    plot_tpg,
    verify_tpg_integrity!,
    TPG_Integrity_Report


const to = TimerOutput()
disable_timer!(to)

# TODO ADD CONDITION THAT A TEAM CAN'T HAVE MULTIPLE TIMES THE SAME PROGRAM

include("programs_and_teams.jl")

##############################################################
# Tangled Program Graphs                                     #
##############################################################

abstract type AbstractTPG end

"""
Holds all programs, all teams and roots.
"""
mutable struct TangledProgramGraph <: AbstractTPG
    id_counter_program::Int
    id_counter_team::Int
    programs::Dict{ProgramID, TPGProgram}
    teams::Dict{TeamID, TPGTeam}
    root_teams::Set{TeamID}
    actions::Vector{Any}

    function TangledProgramGraph()
        return new(0, 0, Dict{ProgramID, TPGProgram}(), Dict{TeamID, TPGTeam}(), Set{TeamID}(), [])
    end
    function TangledProgramGraph(actions::Vector{A}) where {A}
        return new(0, 0, Dict{ProgramID, TPGProgram}(), Dict{TeamID, TPGTeam}(), Set{TeamID}(), actions)
    end
end

"""
    update_actions!(tpg::TangledProgramGraph, new_actions::Vector{Any})
Updates the list of available actions in the TangledProgramGraph.
"""
function update_actions!(tpg::TangledProgramGraph, new_actions::Vector{A}) where {A}
    @warn "Existing program might have actions no longer in new_actions"
    empty!(tpg.actions)
    append!(tpg.actions, new_actions)
    return @info "TPG actions updated to: $(tpg.actions)"
end

include("copy.jl") # copy programs and teams (shallow)
include("cache.jl") # cache, lru

##############################################################
# LOW LEVEL TPG FONCTIONS                                    #
##############################################################

"""
    add_program!(tpg::TangledProgramGraph, genome::UTGenome; action::Union{Any, Nothing} = nothing)::TPGProgram
Adds a new `UTGenome` wrapped in a `TPGProgram` to the `TangledProgramGraph`.
The `UTGenome` is decoded into a `UTCGP.Program` lazily upon first evaluation.
Assigns a unique ID to the new TPGProgram.
If `action` is provided, asserts that it exists in `tpg.actions`.
If `action` is `nothing`, a random action from `tpg.actions` is assigned, or `nothing` if `tpg.actions` is empty.

Warning : in and out edges are not set by this function.
"""
function add_program!(tpg::TangledProgramGraph, genome::UTGenome; action::Union{Any, Nothing} = nothing)::TPGProgram
    tpg.id_counter_program += 1

    assigned_action = nothing
    if action !== nothing
        # Assert that the provided action exists in tpg.actions
        @assert in(action, tpg.actions) "Provided action `$(action)` not found in TPG's available actions."
        assigned_action = action
    elseif !isempty(tpg.actions) # If no action provided, and actions exist, assign a random one
        assigned_action = rand(tpg.actions)
    end

    pid = ProgramID(tpg.id_counter_program)
    tpg_program = TPGProgram(pid, genome, action = assigned_action)
    tpg.programs[pid] = tpg_program
    return tpg_program
end

"""
    add_team!(tpg::TangledProgramGraph, program_ids::Vector{ProgramID}, action_map::Dict{ProgramID, TeamID})::TPGTeam
Adds a new `TPGTeam` to the `TangledProgramGraph`.
`program_ids` are the IDs of `TPGProgram`s that belong to this team.
`action_map` maps a `TPGProgram` ID to the next `TPGTeam` ID.
Assigns a unique ID to the new TPGTeam.

New team's id will be added to all programs in edges.

Outs in `action_map` are set to `out_edges`. For those `out` teams, they respectively receive an `in_edge`
"""
function add_team!(tpg::TangledProgramGraph, program_ids::Vector{ProgramID}, action_map::Dict{ProgramID, TeamID} = Dict{ProgramID, TeamID}())::TPGTeam
    tpg.id_counter_team += 1
    programs = [tpg.programs[id] for id in program_ids]
    tid = TeamID(tpg.id_counter_team)
    tpg_team = TPGTeam(tid, programs, Dict{ProgramID, TeamID}())
    tpg.teams[tid] = tpg_team

    # Update in/out edges
    for pid in program_ids
        push!(tpg.programs[pid].in_edges, tid)
    end

    for (program_id, next_team_id) in action_map
        update_team_action!(tpg, tid, program_id, next_team_id)
    end
    return tpg_team
end

"""
Checks if `program_id` still points to `dest_team_id` in any team *other than* `ignore_team`.
If not, removes `dest_team_id` from the program's `out_edges`.
"""
function _remove_program_out_edge_if_unused!(tpg::TangledProgramGraph, program_id::ProgramID, dest_team_id::TeamID; ignore_team::Union{TeamID, Nothing} = nothing)
    if !haskey(tpg.programs, program_id)
        @warn "Asking to remove out connection from a program that is not in the TPG ? $ProgramID"
        return
    end
    program = tpg.programs[program_id]
    still_used = false
    for owner_team_id in program.in_edges
        if owner_team_id == ignore_team
            continue
        end
        if haskey(tpg.teams, owner_team_id)
            owner_team = tpg.teams[owner_team_id]
            if get(owner_team.action_map, program_id, nothing) == dest_team_id
                still_used = true
                break
            end
        end
    end
    return if !still_used
        delete!(program.out_edges, dest_team_id)
    end
end

"""
Helper to pass only ints for programs and action map
"""
function add_team!(tpg::TangledProgramGraph, program_ids::Vector{Int}, action_map::Dict{Int, Int})::TPGTeam
    # Convert IDs to strong types for internal use
    converted_program_ids = [ProgramID(id) for id in program_ids]
    converted_action_map = Dict{ProgramID, TeamID}()
    for (k, v) in action_map
        converted_action_map[ProgramID(k)] = TeamID(v)
    end

    return add_team!(tpg, converted_program_ids, converted_action_map)
end


"""
    set_root_team!(tpg::TangledProgramGraph, team_id::TeamID)
Sets a team as a root team (entry point) of the Tangled Program Graph.
"""
function set_root_team!(tpg::TangledProgramGraph, team_id::TeamID)
    if !haskey(tpg.teams, team_id)
        error("Team with ID $team_id not found in TPG.")
    end
    return push!(tpg.root_teams, team_id)
end

function set_root_team!(tpg::TangledProgramGraph, team_id::Int)
    return set_root_team!(tpg, TeamID(team_id))
end

include("utilities.jl") # find by ids, resets..
include("traverse.jl") # go from one team to another
include("gc.jl") # remove a team recursively
include("mutations.jl") # mutate teams and programs, "copy on write"
include("evaluation.jl") # eval from root
include("checks.jl")
include("ea.jl")

# include("plot.jl") # TODO
end # module TMAGE
