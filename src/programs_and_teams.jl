##############################################################
# Tangled Program Graphs components                          #
##############################################################

# --- ID Wrappers ---
abstract type AbstractID end
struct ProgramID <: AbstractID
    val::Int
end
Base.hash(x::ProgramID, h::UInt) = hash(x.val, h)
Base.:(==)(x::ProgramID, y::ProgramID) = x.val == y.val
Base.:(==)(x::ProgramID, y::Int) = x.val == y
Base.:(==)(x::Int, y::ProgramID) = x == y.val
Base.show(io::IO, x::ProgramID) = print(io, "P$(x.val)")

struct TeamID <: AbstractID
    val::Int
end
Base.hash(x::TeamID, h::UInt) = hash(x.val, h)
Base.:(==)(x::TeamID, y::TeamID) = x.val == y.val
Base.:(==)(x::TeamID, y::Int) = x.val == y
Base.:(==)(x::Int, y::TeamID) = x == y.val
Base.show(io::IO, x::TeamID) = print(io, "T$(x.val)")


######################
# PROGRAMS AND TEAMS #
######################
abstract type AbstractTPGTeam end

"""
One program (MAGE)

Might be pointed by a team(s)
Might point to another team(s), the path will depend on the teams `action_map`
"""
mutable struct TPGProgram <: AbstractProgram
    id::ProgramID
    genome::UTGenome
    program::Union{Program, Nothing}
    in_edges::Set{TeamID}
    out_edges::Set{TeamID}
    action::Union{Any, Nothing}

    function TPGProgram(id::Int, genome::UTGenome; action::Union{Any, Nothing} = nothing)
        return new(ProgramID(id), genome, nothing, Set{TeamID}(), Set{TeamID}(), action)
    end
    # internal constructor with ID wrapper
    function TPGProgram(id::ProgramID, genome::UTGenome; action::Union{Any, Nothing} = nothing)
        return new(id, genome, nothing, Set{TeamID}(), Set{TeamID}(), action)
    end
end

"""
Holds a bunch of programs that will be evaluated when the `TPGTeam` is itself evaluated to see who wins the bid.

The `action_map` maps a program => the next team. If there is no map, the program is leaf.
"""
mutable struct TPGTeam <: AbstractTPGTeam
    id::TeamID
    programs::Vector{TPGProgram}
    action_map::Dict{ProgramID, TeamID}
    in_edges::Set{TeamID}
    out_edges::Set{TeamID}

    function TPGTeam(id::Int, programs::Vector{TPGProgram}, action_map::Dict{Int, Int})
        # Convert Int dictionary to ID dictionary
        converted_map = Dict{ProgramID, TeamID}()
        for (k, v) in action_map
            converted_map[ProgramID(k)] = TeamID(v)
        end
        return new(TeamID(id), programs, converted_map, Set{TeamID}(), Set{TeamID}())
    end

    # Constructor that accepts already converted action_map (for internal use)
    function TPGTeam(id::TeamID, programs::Vector{TPGProgram}, action_map::Dict{ProgramID, TeamID})
        return new(id, programs, action_map, Set{TeamID}(), Set{TeamID}())
    end
end
