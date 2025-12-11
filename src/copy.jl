# --- Copy Helpers ---

"""
    copy_team(tpg::TangledProgramGraph, team_id::TeamID)::TPGTeam
Creates a shallow copy of a team. The new team has a new ID but shares the same programs and action map (initially).
"""
function copy_team(tpg::TangledProgramGraph, team_id::TeamID)::TPGTeam
    original_team = tpg.teams[team_id]

    # Extract IDs for add_team!
    program_ids = [p.id for p in original_team.programs]
    action_map = copy(original_team.action_map)

    # Use internal add_team helper to register new team
    new_team = add_team!(tpg, program_ids, action_map) # updates connections
    return new_team
end

"""
    copy_program(tpg::TangledProgramGraph, program_id::ProgramID)::TPGProgram
Creates a deep copy of a program. The new program has a new ID and a deep copy of the genome.
"""
function copy_program(tpg::TangledProgramGraph, program_id::ProgramID)::TPGProgram
    original_program = tpg.programs[program_id]

    # Deep copy genome
    new_genome = deepcopy(original_program.genome)

    # Use add_program! to register
    new_program = add_program!(tpg, new_genome, action = original_program.action)
    return new_program
end
