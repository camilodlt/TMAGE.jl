using Revise
using TMAGE
using Test
using UTCGP
using LRUCache
using GraphViz
using Random

function metalib_3int_1float()
    model_arch = modelArchitecture(
        [Int, Int, Int], [1, 1, 1], # one input type int
        [Int, Float64], # two chromosomes. Float is the second
        [Float64], [2] # one output of type float64
    )
    bundles_int = [bundle_integer_basic, bundle_number_arithmetic]
    bundles_int = [deepcopy(b) for b in bundles_int]
    bundles_f = [bundle_float_basic, bundle_number_arithmetic]
    bundles_f = [deepcopy(b) for b in bundles_f]
    update_caster!.(bundles_int, integer_caster)
    update_fallback!.(bundles_int, () -> 0)
    update_caster!.(bundles_f, float_caster)
    update_fallback!.(bundles_f, () -> 0.0)
    return MetaLibrary([Library(bundles_int), Library(bundles_f)]), model_arch
end

"""
(x₁*x₂)/x₃
"""
function sample_program1(ml, ma)
    nc = nodeConfig(10, 1, 3, 3)
    shared_inputs, ut_genome = make_evolvable_utgenome(ma, ml, nc)
    initialize_genome!(ut_genome)

    # node 4 => x1 * x2
    ut_genome[2][4][1].value = argmax(list_functions_names(ml[2]) .== "number_mult")
    ut_genome[2][4][2].value = 1 # connects to x1
    ut_genome[2][4][3].value = 1 # int
    ut_genome[2][4][4].value = 2 # x3
    ut_genome[2][4][5].value = 1

    # node 5 => node 4 / x3
    ut_genome[2][5][1].value = argmax(list_functions_names(ml[2]) .== "number_div")
    ut_genome[2][5][2].value = ut_genome[2][4].x_position
    ut_genome[2][5][3].value = 2
    ut_genome[2][5][4].value = 3 # x3
    ut_genome[2][5][5].value = 1

    # out points to last node (5)
    ut_genome.output_nodes[1][1].value = 1 # identity
    ut_genome.output_nodes[1][2].value = ut_genome[2][5].x_position
    return ut_genome, shared_inputs, ml
end

"""
(x₁/x₂)*x₃
"""
function sample_program2(ml, ma)
    nc = nodeConfig(10, 1, 3, 3)
    shared_inputs, ut_genome = make_evolvable_utgenome(ma, ml, nc)
    initialize_genome!(ut_genome)

    # node 1 => ret 1
    ut_genome[2][1][1].value = argmax(list_functions_names(ml[2]) .== "ret_1")

    # node 4 => x1 * x2
    ut_genome[2][4][1].value = argmax(list_functions_names(ml[2]) .== "number_div")
    ut_genome[2][4][2].value = 1 # connects to x1
    ut_genome[2][4][3].value = 1 # int
    ut_genome[2][4][4].value = 2 # x3
    ut_genome[2][4][5].value = 1

    # node 5 => node 4 / x3
    ut_genome[2][5][1].value = argmax(list_functions_names(ml[2]) .== "number_mult")
    ut_genome[2][5][2].value = ut_genome[2][4].x_position
    ut_genome[2][5][3].value = 2
    ut_genome[2][5][4].value = 3 # x3
    ut_genome[2][5][5].value = 1

    # out points to last node (5)
    ut_genome.output_nodes[1][1].value = 1 # identity
    ut_genome.output_nodes[1][2].value = ut_genome[2][5].x_position
    return ut_genome, shared_inputs, ml
end

"""
x1*x2
"""
function sample_program3(ml, ma)
    nc = nodeConfig(10, 1, 3, 3)
    shared_inputs, ut_genome = make_evolvable_utgenome(ma, ml, nc)
    initialize_genome!(ut_genome)

    # node 4 => x1 * x2
    ut_genome[2][4][1].value = argmax(list_functions_names(ml[2]) .== "number_mult")
    ut_genome[2][4][2].value = 1 # connects to x1
    ut_genome[2][4][3].value = 1 # int
    ut_genome[2][4][4].value = 2 # x3
    ut_genome[2][4][5].value = 1

    # out points to last node (5)
    ut_genome.output_nodes[1][1].value = 1 # identity
    ut_genome.output_nodes[1][2].value = ut_genome[2][4].x_position
    return ut_genome, shared_inputs, ml
end

@testset "TMAGE.jl" begin
    @testset "TPG creation no actions" begin
        tpg = TangledProgramGraph()
        @test tpg isa TMAGE.TangledProgramGraph
        @test isempty(tpg.programs)
        @test isempty(tpg.teams)
        @test isempty(tpg.root_teams)
        @test tpg.actions |> isempty
    end
    @testset "TPG creation actions" begin
        actions = [1, 2, 3]
        tpg = TangledProgramGraph(actions)
        @test tpg isa TMAGE.TangledProgramGraph
        @test isempty(tpg.programs)
        @test isempty(tpg.teams)
        @test isempty(tpg.root_teams)
    end

    @testset "Add programs & Teams to TPG" begin
        actions = [1, 2, 3]
        tpg = TangledProgramGraph(actions)
        ml, ma = metalib_3int_1float()
        prog1, si, _ = sample_program1(ml, ma)
        prog2, _, _ = sample_program2(ml, ma)
        prog3, _, _ = sample_program3(ml, ma)
        nc = nodeConfig(10, 1, 3, 3)
        tpg_program1 = TMAGE._add_program!(tpg, prog1)
        tpg_program2 = TMAGE._add_program!(tpg, prog2)
        tpg_program3 = TMAGE._add_program!(tpg, prog3)

        @test tpg_program1.id == 1
        @test tpg_program1.program === nothing

        @test tpg_program2.id == 2
        @test tpg_program2.program === nothing

        @test tpg_program3.id == 3
        @test tpg_program2.program === nothing

        @test tpg.id_counter_program == 3
        @test tpg.programs |> length == 3

        # ADD TEAMS
        # # add team 1
        team1 = TMAGE._add_team!(tpg, [tpg_program1.id, tpg_program2.id], Dict{Int, Int}())

        # # add team 2
        action_map1 = Dict(1 => 1) # if p1 wins => go to team 1. if p2 wins => no next
        team2 = TMAGE._add_team!(tpg, [tpg_program1.id, tpg_program2.id], action_map1)
        @test team2.out_edges |> length == 1
        @test team2.action_map == action_map1
        @test team2.in_edges |> length == 0
        @test team2.id == 2
        @test !haskey(team2.action_map, tpg_program2.id) # prog 2 does not point

        # # add team 3
        action_map2 = Dict{Int, Int}() # no pointers, "leaf" team
        team3 = TMAGE._add_team!(tpg, [tpg_program3.id], action_map2)
        @test team3.out_edges |> length == 0
        @test team3.action_map |> isempty

        # check edges
        @test tpg_program1.in_edges == Set([team1.id, team2.id]) # Team 1,2 uses prog1
        @test tpg_program2.in_edges == Set([team1.id, team2.id]) # Team 1,2 uses prog2
        @test tpg_program3.in_edges == Set([team3.id]) # Team 3 uses prog3

        @test team1.out_edges == Set()
        @test team1.in_edges == Set(2)
        @test team2.out_edges == Set([team1.id]) # team 2 can go to team 1 via prog 1


        TMAGE.set_root_team!(tpg, team2.id)
        @test tpg.root_teams == Set([team2.id])
    end

    @testset "TPG eval with actions" begin
        actions = [1, 2, 3]
        tpg = TangledProgramGraph(actions)
        ml, ma = metalib_3int_1float()
        prog1, si, _ = sample_program1(ml, ma)
        prog2, _, _ = sample_program2(ml, ma)
        prog3, _, _ = sample_program3(ml, ma)
        nc = nodeConfig(10, 1, 3, 3)
        tpg_program1 = TMAGE._add_program!(tpg, prog1)
        tpg_program2 = TMAGE._add_program!(tpg, prog2)
        tpg_program3 = TMAGE._add_program!(tpg, prog3)

        # ADD TEAMS
        # # add team 1
        team1 = TMAGE._add_team!(tpg, [tpg_program1.id, tpg_program2.id], Dict{Int, Int}())

        # # add team 2
        action_map1 = Dict(1 => 1) # if p1 wins => go to team 1. if p2 wins => no next
        team2 = TMAGE._add_team!(tpg, [tpg_program1.id, tpg_program2.id], action_map1)

        # # add team 3
        action_map2 = Dict{Int, Int}() # no pointers, "leaf" team
        team3 = TMAGE._add_team!(tpg, [tpg_program3.id], action_map2)

        TMAGE.set_root_team!(tpg, team2.id)
        @test tpg.root_teams == Set([team2.id])
        TMAGE.set_root_team!(tpg, team3.id)
        @test tpg.root_teams == Set((2, 3))


        input_data = [1, 2, 3]
        output, path = evaluate(tpg, 2, input_data, si, ml, ma; cache_mode = PerInputCache)
        # root 2 = team 2
        # - prog 1 => 1*2/3 = 0.666
        # - prog 2 => 1/2*3 = 1.5 => wins & leaf so output = tpg_program2.action
        @test output == tpg_program2.action


        input_data = [1, 2, 1]
        output, path = evaluate(tpg, 2, input_data, si, ml, ma; cache_mode = PerInputCache)
        # root 2 = team 2
        # - prog 1 => 1*2/1 = 2 => wins  => points to team1
        # - prog 2 => 1/2*3 = 0.5
        # - - team 1
        # - - - prog1 => 2 => wins && leaf => tpg_program1.action
        # - - - prog2 => 0.5
        @test output == tpg_program1.action

    end
end

@testset "CleanUp" begin
    ml, ma = metalib_3int_1float()
    nc = nodeConfig(10, 1, 3, 3)
    prog1, si, _ = sample_program1(ml, ma) #(x₁*x₂)/x₃
    prog2, _, _ = sample_program2(ml, ma) #(x₁/x₂)*x₃
    prog3, _, _ = sample_program3(ml, ma) #x1*x2
    @test begin
        tpg = TangledProgramGraph([1, 2])
        tpg_program1 = TMAGE.add_program!(tpg, prog1)
        tpg_program2 = TMAGE.add_program!(tpg, prog2)
        tpg_program3 = TMAGE.add_program!(tpg, prog3)

        team1 = TMAGE.add_team!(tpg, [tpg_program1.id, tpg_program2.id], Dict{ProgramID, TeamID}())
        set_root_team!(tpg, team1.id)
        verify_tpg_integrity!(tpg, cleanup_orphans = true)
        tpg.programs |> length == 2 # removes prog 3 with no parent
    end

    @test begin
        tpg = TangledProgramGraph([1, 2])
        tpg_program1 = TMAGE.add_program!(tpg, prog1)
        tpg_program2 = TMAGE.add_program!(tpg, prog2)
        tpg_program3 = TMAGE.add_program!(tpg, prog3)

        team1 = TMAGE.add_team!(tpg, [tpg_program1.id, tpg_program2.id])
        team2 = TMAGE.add_team!(tpg, [tpg_program2.id])
        team3 = TMAGE.add_team!(tpg, [tpg_program3.id])
        team4 = TMAGE.add_team!(tpg, [tpg_program2.id, tpg_program3.id])
        set_root_team!(tpg, team1.id)
        set_root_team!(tpg, team3.id)

        update_team_action!(tpg, team1.id, tpg_program1.id, team2.id) # 1 => 2 via P1
        update_team_action!(tpg, team3.id, tpg_program3.id, team4.id) # 3 => 4 via P3

        r = verify_tpg_integrity!(tpg)
        @test r.is_consistent == true # all teams and programs are reachable

        # if we remove team 3
        # T3 and T4 are orphan
        # P3 is also orphan
        # P2 although used by T4 is not orphan because T1&T2 use it
        delete!(tpg.root_teams, team3.id)

        r = verify_tpg_integrity!(tpg)
        @test r.is_consistent == false
        @test r.coverage_teams_percent == 50.0 # T1, T2 ok || T3, T4 nok
        @test r.coverage_programs_percent == (2 / 3) * 100 # P1,P2 used || P3 not used
        @test r.orphaned_programs == Set((tpg_program3.id,))
        @test r.orphaned_teams == Set((team3.id, team4.id))
        @test r.reachable_teams_count == 2

        # if we gc
        r = verify_tpg_integrity!(tpg; cleanup_orphans = true)
        @test r.is_consistent == true
        @test r.orphaned_teams |> isempty
        @test r.orphaned_programs |> isempty
        @test keys(tpg.teams) == Set((TeamID(1), TeamID(2)))
        @test keys(tpg.programs) == Set((ProgramID(1), ProgramID(2)))
    end

    @test begin # opposite as previous test
        tpg = TangledProgramGraph([1, 2])
        tpg_program1 = TMAGE.add_program!(tpg, prog1)
        tpg_program2 = TMAGE.add_program!(tpg, prog2)
        tpg_program3 = TMAGE.add_program!(tpg, prog3)

        team1 = TMAGE.add_team!(tpg, [tpg_program1.id, tpg_program2.id])
        team2 = TMAGE.add_team!(tpg, [tpg_program2.id])
        team3 = TMAGE.add_team!(tpg, [tpg_program3.id])
        team4 = TMAGE.add_team!(tpg, [tpg_program2.id, tpg_program3.id])
        set_root_team!(tpg, team1.id)
        set_root_team!(tpg, team3.id)

        update_team_action!(tpg, team1.id, tpg_program1.id, team2.id) # 1 => 2 via P1
        update_team_action!(tpg, team3.id, tpg_program3.id, team4.id) # 3 => 4 via P3

        r = verify_tpg_integrity!(tpg)
        @test r.is_consistent == true # all teams and programs are reachable

        # if we remove team 1
        # T1 and T2 are orphan
        # P1 is also orphan
        # P2 although used by T1,T2 is not orphan because T3 use it
        delete!(tpg.root_teams, team1.id)

        r = verify_tpg_integrity!(tpg)
        @test r.is_consistent == false
        @test r.coverage_teams_percent == 50.0 # T1, T2 ok || T3, T4 nok
        @test r.coverage_programs_percent == (2 / 3) * 100 # P1,P2 used || P3 not used
        @test r.orphaned_programs == Set((tpg_program1.id,))
        @test r.orphaned_teams == Set((team1.id, team2.id))
        @test r.reachable_teams_count == 2

        # if we gc
        r = verify_tpg_integrity!(tpg; cleanup_orphans = true)
        @test r.is_consistent == true
        @test r.orphaned_teams |> isempty
        @test r.orphaned_programs |> isempty
        @test keys(tpg.teams) == Set((TeamID(3), TeamID(4)))
        @test keys(tpg.programs) == Set((ProgramID(2), ProgramID(3)))
    end
end

@testset "TPGMutationStrategy" begin
    # program mutation # OK
    # team program addition
    # program action mutation
    # team action map change
    # team program removal

    Random.seed!(789)
    ml, ma = metalib_3int_1float()
    nc = nodeConfig(10, 1, 3, 3)
    si, _ = make_evolvable_utgenome(ma, ml, nc)
    prog1_genome, _, _ = sample_program1(ml, ma) # (x1*x2)/x3
    prog2_genome, _, _ = sample_program2(ml, ma) # (x1/x2)*x3
    prog3_genome, _, _ = sample_program3(ml, ma) # (x1*x2)

    @testset "Program mutation" begin # the new root had a mutation in one of the programs. => New mutated program, rest stays equal
        tpg = TangledProgramGraph([1, 2, 3])
        p1 = TMAGE.add_program!(tpg, prog1_genome, action = 1) # Assign action 1
        team = TMAGE.add_team!(tpg, [p1.id])
        team2 = TMAGE.add_team!(tpg, [p1.id])
        update_team_action!(tpg, team.id, p1.id, team2.id) # 1 => 2 via P1
        set_root_team!(tpg, team.id)

        # Store original team and program states for comparison
        original_tpg_programs_count = length(tpg.programs)
        original_tpg_teams_count = length(tpg.teams)
        original_team_id = team.id
        original_team_programs_ids = Set([p.id for p in team.programs])
        original_team_action_map = copy(team.action_map)
        original_program_genome = deepcopy(p1.genome) # Deep copy the genome for comparison
        input_data = [4, 5, 2] # Sample input: (x1=4, x2=5, x3=2)
        original_team_output_before_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache) # outputs 1 since its the action and only program in team

        # MUTATION
        config = TPGMutationConfig(
            1.0, # program_mutation_rate (FORCE MUTATE NODE)
            0.0, # add_program_to_team_rate
            0.0, # remove_program_from_team_rate
            0.0, # mutate_action_map_rate
            0.0  # mutate_program_action_rate
        )
        strategy = TPGMutationStrategy()
        TMAGE._mutate_single_offspring!(tpg, original_team_id, strategy, config, ma, ml, nc, si)

        # The original TPG should have one more team and one more program (the mutated copy)
        @test length(tpg.teams) == original_tpg_teams_count + 1
        @test length(tpg.programs) == original_tpg_programs_count + 1 # Original program + mutated copy

        # Retrieve the original team (by its ID, which should still exist)
        retrieved_original_team = find_team_by_id(tpg, original_team_id)
        @test retrieved_original_team !== nothing
        @test Set([p.id for p in retrieved_original_team.programs]) == original_team_programs_ids
        @test retrieved_original_team.action_map == original_team_action_map
        @test UTCGP.general_hasher_sha(p1.genome) == UTCGP.general_hasher_sha(original_program_genome) # P1 for original team unchanged

        # Check the newly created (mutated) team
        new_team_id = setdiff(keys(tpg.teams), Set([original_team_id, TeamID(2)])) |> first
        new_team = find_team_by_id(tpg, new_team_id)
        @test new_team !== nothing
        @test length(new_team.programs) == length(original_team_programs_ids) # Should have the same number of programs
        @test Set(values(new_team.action_map)) == Set(values(original_team_action_map)) # they map to the same team
        @test Set(keys(original_team_action_map)) == Set([p1.id]) # but by via diff programs
        @test Set(keys(new_team.action_map)) == Set([ProgramID(2)]) # but by via diff programs

        # Identify the mutated program within the new team
        new_program_ids_in_new_team = Set([p.id for p in new_team.programs])
        # The new team should have a program with an ID different from the original program
        mutated_program_id = setdiff(new_program_ids_in_new_team, original_team_programs_ids) |> first
        mutated_program = find_program_by_id(tpg, mutated_program_id)
        @test mutated_program !== nothing
        @test mutated_program_id.val > original_tpg_programs_count # New program ID should be higher

        # The mutated program's genome should be different from the original program's genome
        @test UTCGP.general_hasher_sha(mutated_program.genome) != UTCGP.general_hasher_sha(original_program_genome)

        # --- Evaluation after mutation ---
        original_team_output_after_mutation, orig_path = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_after_mutation == original_team_output_before_mutation # Original team output must be the same

        new_team_output, new_path = evaluate(tpg, new_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test new_team_output == original_team_output_before_mutation # output still the same since only one action ...
        @test new_path[1] != orig_path[1] # eval of first program diff
        @test new_path[2] == orig_path[2] # but since they both point to T2, the remaining path is the same

        # --- Integrity Check ---
        report = verify_tpg_integrity!(tpg; cleanup_orphans = false)
        @test report.is_consistent # Ensure no inconsistencies introduced
        @test isempty(report.orphaned_teams)
        @test isempty(report.orphaned_programs)
    end
    @testset "Add Program" begin # We add a new program to the new root.
        tpg = TangledProgramGraph([1, 2, 3])
        p1 = TMAGE.add_program!(tpg, prog1_genome, action = 1) # Assign action 1
        p2 = TMAGE.add_program!(tpg, prog2_genome, action = 2) # Assign action 2 # program not in any team, so we are sure it can be added to our new root
        team = TMAGE.add_team!(tpg, [p1.id])
        team2 = TMAGE.add_team!(tpg, [p1.id])
        update_team_action!(tpg, team.id, p1.id, team2.id) # 1 => 2 via P1
        set_root_team!(tpg, team.id)

        # Store original team and program states for comparison
        original_tpg_programs_count = length(tpg.programs)
        original_tpg_teams_count = length(tpg.teams)
        original_team_id = team.id
        original_team_programs_ids = Set([p.id for p in team.programs])
        original_team_action_map = copy(team.action_map)
        original_program_genome = deepcopy(p1.genome) # Deep copy the genome for comparison
        input_data = [4, 5, 2] # Sample input: (x1=4, x2=5, x3=2)
        original_team_output_before_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache) # outputs 1 since its the action and only program in team

        # MUTATION
        config = TPGMutationConfig(
            0.0, # program_mutation_rate (FORCE MUTATE NODE)
            1.0, # add_program_to_team_rate
            0.0, # remove_program_from_team_rate
            0.0, # mutate_action_map_rate
            0.0  # mutate_program_action_rate
        )
        strategy = TPGMutationStrategy()
        TMAGE._mutate_single_offspring!(tpg, original_team_id, strategy, config, ma, ml, nc, si)
        # [ Info: Added new program P2 to team T3 . Present action : 2.

        # The original TPG should have one more team and one more program (the mutated copy)
        @test length(tpg.teams) == original_tpg_teams_count + 1
        @test length(tpg.programs) == original_tpg_programs_count # one program was added to the new root, but an existing program so the set of programs is the same

        # Retrieve the original team (by its ID, which should still exist)
        retrieved_original_team = find_team_by_id(tpg, original_team_id)
        @test retrieved_original_team !== nothing
        @test Set([p.id for p in retrieved_original_team.programs]) == original_team_programs_ids
        @test retrieved_original_team.action_map == original_team_action_map
        @test UTCGP.general_hasher_sha(p1.genome) == UTCGP.general_hasher_sha(original_program_genome) # P1 for original team unchanged
        @test length(team.programs) == 1

        # Check the newly created (mutated) team
        new_team_id = setdiff(keys(tpg.teams), Set([original_team_id, TeamID(2)])) |> first
        new_team = find_team_by_id(tpg, new_team_id)
        @test new_team !== nothing
        @test length(new_team.programs) == length(original_team_programs_ids) + 1 # Has an extra program
        @test new_team.action_map[ProgramID(1)] == TeamID(2) # as the parent team
        @test !(haskey(new_team.action_map, ProgramID(2)))

        # --- Evaluation after mutation ---
        original_team_output_after_mutation, orig_path = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_after_mutation == original_team_output_before_mutation # Original team output must be the same
        # output is action 1

        new_team_output, new_path = evaluate(tpg, new_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        # p1 => 10. p2 => 1.6 => P1 WINS bid => t2 => p1 => action 1
        @test new_team_output == 1

        # --- Integrity Check ---
        report = verify_tpg_integrity!(tpg; cleanup_orphans = false)
        @test report.is_consistent # Ensure no inconsistencies introduced
        @test isempty(report.orphaned_teams)
        @test isempty(report.orphaned_programs)
    end
    @testset "Remove Program From Team" begin
        Random.seed!(123)
        actions = [1, 2, 3]
        tpg = TangledProgramGraph(actions)

        p1 = TMAGE.add_program!(tpg, prog1_genome, action = 1)
        p2 = TMAGE.add_program!(tpg, prog2_genome, action = 2)
        p3 = TMAGE.add_program!(tpg, prog3_genome, action = 3)

        p4 = TMAGE.add_program!(tpg, prog3_genome, action = 1)
        out_team = TMAGE.add_team!(tpg, [p4.id])

        # Create original_team with p1, p2, p3
        original_team_progs = [p1.id, p2.id, p3.id]
        original_team_action_map = Dict(p2.id => out_team.id)
        original_team = TMAGE.add_team!(tpg, original_team_progs, original_team_action_map)
        set_root_team!(tpg, original_team.id)

        # Store relevant original TPG and component states *before* mutation for comparison
        original_tpg_programs_count = length(tpg.programs)
        original_tpg_teams_count = length(tpg.teams)

        original_team_id = original_team.id
        original_team_programs_set = Set([p.id for p in original_team.programs])
        original_team_action_map_copy = deepcopy(original_team.action_map)
        original_team_out_edges_copy = deepcopy(original_team.out_edges)
        original_p2_in_edges = deepcopy(p2.in_edges)

        # --- Evaluation before mutation ---
        input_data = [4, 5, 2]
        # Original Team behavior with input_data:
        # p1: 10.0 (action 1)
        # p2: 1.6 (action 2)
        # p3: 20.0 (action 3)
        # Winner is p3. p3 is NOT in original_team_action_map. So, output is p3.action = 3.
        original_team_output_before_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_before_mutation == 3

        # 2. Configure Mutation: Force 'remove_program_from_team_rate'
        config = TPGMutationConfig(
            0.0, # program_mutation_rate
            0.0, # add_program_to_team_rate
            1.0, # remove_program_from_team_rate
            0.0, # mutate_action_map_rate
            0.0  # mutate_program_action_rate

        )
        strategy = TPGMutationStrategy()

        # 3. Perform mutation
        # This will create new_team (a copy of original_team) and then remove a program from new_team.
        TMAGE._mutate_single_offspring!(tpg, original_team_id, strategy, config, ma, ml, nc, si)
        # WITH SEED 123 REMOVES P2 FROM T3

        # 4. Assertions
        # TPG global counts
        @test length(tpg.teams) == original_tpg_teams_count + 1 # One new team (the copy)
        @test length(tpg.programs) == original_tpg_programs_count # No global programs added/removed

        # Retrieve the original team (by its ID) - it must be untouched internally
        retrieved_original_team = find_team_by_id(tpg, original_team_id)
        @test retrieved_original_team !== nothing
        @test Set([p.id for p in retrieved_original_team.programs]) == original_team_programs_set
        @test retrieved_original_team.action_map == original_team_action_map_copy
        @test retrieved_original_team.out_edges == original_team_out_edges_copy

        # Check the newly created (mutated) team
        new_team_id = setdiff(keys(tpg.teams), Set([original_team_id])) |> first
        new_team = find_team_by_id(tpg, new_team_id)
        @test new_team !== nothing
        @test length(new_team.programs) == length(original_team_programs_set) - 1 # One program removed

        # Identify the program that was removed from the new team
        removed_program_id = setdiff(original_team_programs_set, Set([p.id for p in new_team.programs])) |> first
        @test removed_program_id == ProgramID(2)
        removed_program = find_program_by_id(tpg, removed_program_id)
        @test removed_program !== nothing
        @test collect(removed_program.in_edges) == [original_team.id] # only orig team uses it not the new one

        # Verify new_team's programs and action map
        @test !(removed_program_id in [p.id for p in new_team.programs]) # Removed program not in new_team's programs
        @test !haskey(new_team.action_map, removed_program_id) # Removed program not in new_team's action_map

        # New_team's action map should be a subset of original's, excluding removed program's entry
        @test new_team.action_map |> isempty # Since only p2 was pointing to another team

        # --- Verify in_edges for programs ---
        @test Set([original_team_id, new_team_id]) == p1.in_edges # p1 is in both original and new team
        @test Set([original_team_id]) == p2.in_edges # p2 removed from new
        @test Set([original_team_id, new_team_id]) == p3.in_edges # p3 in both

        # --- Evaluation after mutation ---
        original_team_output_after_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_after_mutation == original_team_output_before_mutation # Original team output must be the same

        # New Team behavior with input_data:
        # Now new_team has p1 and p2. p3 was removed.
        # p1: 10.0 (action 1)
        # p3: 20.0 (action 3)
        # winner is p3. Since no action map. p3 wins => action 3
        new_team_output, path = evaluate(tpg, new_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test new_team_output == 3

        # --- Integrity Check ---
        report = verify_tpg_integrity!(tpg; cleanup_orphans = false)
        @test report.is_consistent # Ensure no inconsistencies introduced
        @test isempty(report.orphaned_teams)
        @test isempty(report.orphaned_programs)
    end

    @testset "TPGMutationStrategy - mutate_action_map_rate => Removes" begin
        Random.seed!(111)
        tpg = TangledProgramGraph([1, 2, 3])
        p1 = TMAGE.add_program!(tpg, prog1_genome, action = 1)
        p2 = TMAGE.add_program!(tpg, prog2_genome, action = 2)
        team_target1 = TMAGE.add_team!(tpg, [p1.id])
        team_target2 = TMAGE.add_team!(tpg, [p2.id])
        set_root_team!(tpg, team_target2.id) # a root

        # Create the original team that will be copied and mutated
        # p1 points to team_target1. p2 is just a program in the team.
        original_team_action_map = Dict(p1.id => team_target1.id)
        original_team_progs = [p1.id, p2.id]
        original_team = TMAGE.add_team!(tpg, original_team_progs, original_team_action_map)
        set_root_team!(tpg, original_team.id)

        # ORIGINAL
        original_tpg_programs_count = length(tpg.programs)
        original_tpg_teams_count = length(tpg.teams)

        original_team_id = original_team.id
        original_team_programs_set = Set([p.id for p in original_team.programs])
        original_team_action_map_copy = deepcopy(original_team.action_map)
        original_team_out_edges_copy = deepcopy(original_team.out_edges)

        original_p1_out_edges = deepcopy(p1.out_edges) # Should contain original_team_target1.id
        original_p2_in_edges = deepcopy(p2.in_edges) # Should contain original_team_id

        # --- Evaluation before mutation ---
        input_data = [4, 5, 2]
        # Original Team behavior with input_data:
        # p1: 10.0 (action 1)
        # p2: 1.6 (action 2)
        # Winner is p1 (10.0). p1 points to team_target1.
        # Evaluation will proceed to team_target1.
        # team_target1 contains only p1. p1's action is 1. So, final output is 1.
        original_team_output_before_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_before_mutation == 1

        # 2. Configure Mutation: Force 'mutate_action_map_rate'
        config = TPGMutationConfig(
            0.0, # program_mutation_rate
            0.0, # add_program_to_team_rate
            0.0, # remove_program_from_team_rate
            1.0, # mutate_action_map_rate (FORCE MUTATE ACTION MAP)
            0.0  # mutate_program_action_rate
        )
        strategy = TPGMutationStrategy()

        # 3. Perform mutation
        TMAGE._mutate_single_offspring!(tpg, original_team_id, strategy, config, ma, ml, nc, si)

        # 4. Assertions
        @test length(tpg.teams) == original_tpg_teams_count + 1 # One new team (the copy)
        @test length(tpg.programs) == original_tpg_programs_count # No global programs added/removed

        # Retrieve the original team (by its ID) - it must be untouched internally
        retrieved_original_team = find_team_by_id(tpg, original_team_id)
        @test retrieved_original_team !== nothing
        @test Set([p.id for p in retrieved_original_team.programs]) == original_team_programs_set
        @test retrieved_original_team.action_map == original_team_action_map_copy
        @test retrieved_original_team.out_edges == original_team_out_edges_copy

        # Check the newly created (mutated) team
        new_team_id = setdiff(keys(tpg.teams), Set([original_team_id])) |> first
        new_team = find_team_by_id(tpg, new_team_id)
        @test new_team !== nothing
        @test Set([p.id for p in new_team.programs]) == original_team_programs_set # Same programs

        # Identify the mutated action map entry:
        @test new_team.action_map |> isempty # The mapping for p1 was removed since it was the only one

        original_team_output_after_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_after_mutation == original_team_output_before_mutation # Original team output must be the same

        # out for new team is action 1 since p1 is winner and no maps
        new_team_out, _ = evaluate(tpg, new_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test new_team_out == 1

        # Test program in edges
        @test p1.in_edges == Set((TeamID(1), TeamID(3), TeamID(4))) # new team has it
        # Test program out edges
        @test p1.out_edges == original_p1_out_edges # unchanged

        # New team out edge is empty
        @test new_team.out_edges |> isempty

        # --- Integrity Check ---
        report = verify_tpg_integrity!(tpg; cleanup_orphans = false)
        @test report.is_consistent # Ensure no inconsistencies introduced
        @test isempty(report.orphaned_teams)
        @test isempty(report.orphaned_programs)
    end

    @testset "TPGMutationStrategy - mutate_action_map_rate => Change" begin
        Random.seed!(234)
        tpg = TangledProgramGraph([1, 2, 3])
        p1 = TMAGE.add_program!(tpg, prog1_genome, action = 1)
        p2 = TMAGE.add_program!(tpg, prog2_genome, action = 2)
        team_target1 = TMAGE.add_team!(tpg, [p1.id])
        team_target2 = TMAGE.add_team!(tpg, [p2.id])
        set_root_team!(tpg, team_target2.id) # a root

        # Create the original team that will be copied and mutated
        # p1 points to team_target1. p2 is just a program in the team.
        original_team_action_map = Dict(p1.id => team_target1.id)
        original_team_progs = [p1.id, p2.id]
        original_team = TMAGE.add_team!(tpg, original_team_progs, original_team_action_map)
        set_root_team!(tpg, original_team.id)

        # ORIGINAL
        original_tpg_programs_count = length(tpg.programs)
        original_tpg_teams_count = length(tpg.teams)

        original_team_id = original_team.id
        original_team_programs_set = Set([p.id for p in original_team.programs])
        original_team_action_map_copy = deepcopy(original_team.action_map)
        original_team_out_edges_copy = deepcopy(original_team.out_edges)

        # --- Evaluation before mutation ---
        input_data = [4, 5, 2]
        # Original Team behavior with input_data:
        # p1: 10.0 (action 1)
        # p2: 1.6 (action 2)
        # Winner is p1 (10.0). p1 points to team_target1.
        # Evaluation will proceed to team_target1.
        # team_target1 contains only p1. p1's action is 1. So, final output is 1.
        original_team_output_before_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_before_mutation == 1

        # 2. Configure Mutation: Force 'mutate_action_map_rate'
        config = TPGMutationConfig(
            0.0, # program_mutation_rate
            0.0, # add_program_to_team_rate
            0.0, # remove_program_from_team_rate
            1.0, # mutate_action_map_rate (FORCE MUTATE ACTION MAP)
            0.0  # mutate_program_action_rate
        )
        strategy = TPGMutationStrategy()

        # 3. Perform mutation
        TMAGE._mutate_single_offspring!(tpg, original_team_id, strategy, config, ma, ml, nc, si)

        # New state
        @test length(tpg.teams) == original_tpg_teams_count + 1 # One new team (the copy)
        @test length(tpg.programs) == original_tpg_programs_count # No global programs added/removed

        # Original team is unchanged
        retrieved_original_team = find_team_by_id(tpg, original_team_id)
        @test retrieved_original_team !== nothing
        @test Set([p.id for p in retrieved_original_team.programs]) == original_team_programs_set
        @test retrieved_original_team.action_map == original_team_action_map_copy
        @test retrieved_original_team.out_edges == original_team_out_edges_copy

        # Check the newly created (mutated) team
        new_team_id = setdiff(keys(tpg.teams), Set([original_team_id])) |> first
        new_team = find_team_by_id(tpg, new_team_id)
        @test new_team !== nothing
        @test Set([p.id for p in new_team.programs]) == original_team_programs_set # Same programs

        # Identify the mutated action map entry:
        @test !(new_team.action_map |> isempty) # the mapping was changed
        @test new_team.action_map[collect(keys(original_team_action_map))[1]] != collect(values(original_team_action_map))[1] # the program points to a diff team

        original_team_output_after_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_after_mutation == original_team_output_before_mutation # Original team output must be the same

        # Out for new team is : p1 wins bid and points to t3 (original)
        # original has p1, p2. p1 wins bid.
        # points to t1 which has only p1 => action 1
        new_team_out, _ = evaluate(tpg, new_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test new_team_out == 1

        # Test program in edges
        @test p1.in_edges == Set((TeamID(1), TeamID(3), TeamID(4))) # new team has it
        # Test program out edges
        @test p1.out_edges == Set([TeamID(1), TeamID(3)]) # T1 because of orig team, t3 because of new team

        # New team out edge is empty
        @test new_team.out_edges == Set((TeamID(3),))

        # --- Integrity Check ---
        report = verify_tpg_integrity!(tpg; cleanup_orphans = false)
        @test report.is_consistent # Ensure no inconsistencies introduced
        @test isempty(report.orphaned_teams)
        @test isempty(report.orphaned_programs)
    end

    @testset "TPGMutationStrategy - mutate_action_map_rate => Add" begin
        Random.seed!(222)
        tpg = TangledProgramGraph([1, 2, 3])
        p1 = TMAGE.add_program!(tpg, prog1_genome, action = 1)
        p2 = TMAGE.add_program!(tpg, prog2_genome, action = 2)
        team_target1 = TMAGE.add_team!(tpg, [p1.id])
        team_target2 = TMAGE.add_team!(tpg, [p2.id])
        set_root_team!(tpg, team_target2.id) # a root

        # Create the original team that will be copied and mutated
        # p1 points to team_target1. p2 is just a program in the team.
        original_team_action_map = Dict(p1.id => team_target1.id)
        original_team_progs = [p1.id, p2.id]
        original_team = TMAGE.add_team!(tpg, original_team_progs, original_team_action_map)
        set_root_team!(tpg, original_team.id)

        # ORIGINAL
        original_tpg_programs_count = length(tpg.programs)
        original_tpg_teams_count = length(tpg.teams)

        original_team_id = original_team.id
        original_team_programs_set = Set([p.id for p in original_team.programs])
        original_team_action_map_copy = deepcopy(original_team.action_map)
        original_team_out_edges_copy = deepcopy(original_team.out_edges)

        # --- Evaluation before mutation ---
        input_data = [1, 1, 10]
        # Original Team behavior with input_data:
        # p1: 0.1 (action 1)
        # p2: 10. (action 2)
        # Winner is p2. p2 => action 2
        original_team_output_before_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_before_mutation == 2

        # 2. Configure Mutation: Force 'mutate_action_map_rate'
        config = TPGMutationConfig(
            0.0, # program_mutation_rate
            0.0, # add_program_to_team_rate
            0.0, # remove_program_from_team_rate
            1.0, # mutate_action_map_rate (FORCE MUTATE ACTION MAP)
            0.0  # mutate_program_action_rate
        )
        strategy = TPGMutationStrategy()

        # 3. Perform mutation
        TMAGE._mutate_single_offspring!(tpg, original_team_id, strategy, config, ma, ml, nc, si)
        # [ Info: Action map: Added mapping for program P2 to team T2 in team T4.

        # New state
        @test length(tpg.teams) == original_tpg_teams_count + 1 # One new team (the copy)
        @test length(tpg.programs) == original_tpg_programs_count # No global programs added/removed

        # Original team is unchanged
        retrieved_original_team = find_team_by_id(tpg, original_team_id)
        @test retrieved_original_team !== nothing
        @test Set([p.id for p in retrieved_original_team.programs]) == original_team_programs_set
        @test retrieved_original_team.action_map == original_team_action_map_copy
        @test retrieved_original_team.out_edges == original_team_out_edges_copy

        # Check the newly created (mutated) team
        new_team_id = setdiff(keys(tpg.teams), Set([original_team_id])) |> first
        new_team = find_team_by_id(tpg, new_team_id)
        @test new_team !== nothing
        @test Set([p.id for p in new_team.programs]) == original_team_programs_set # Same programs

        # Identify the mutated action map entry:
        # p1 => t1 (as orig). p2 => something (new)
        @test length(new_team.action_map) == 2
        @test new_team.action_map[ProgramID(1)] == TeamID(1)
        @test new_team.action_map[ProgramID(2)] == TeamID(2)

        original_team_output_after_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_after_mutation == original_team_output_before_mutation # Original team output must be the same

        # p1 = 0.1, p2 = 10. P2 wins bid
        # p2 points to T2 which only has p2 => action 2
        new_team_out, path = evaluate(tpg, new_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test new_team_out == 2
        @test length(path) == 2

        # Test program in edges
        @test p1.in_edges == Set((TeamID(1), TeamID(3), TeamID(4))) # new team has it
        @test p2.in_edges == Set((TeamID(2), TeamID(3), TeamID(4))) # new team has it

        # Test program out edges
        @test p1.out_edges == Set([TeamID(1)]) # T1 because of orig team
        @test p2.out_edges == Set([TeamID(2)]) # new team uses p2 => t2

        # New team out edge is empty
        @test new_team.out_edges == Set((TeamID(1), TeamID(2))) # p1=>t1, p2=>t2

        # --- Integrity Check ---
        report = verify_tpg_integrity!(tpg; cleanup_orphans = false)
        @test report.is_consistent # Ensure no inconsistencies introduced
        @test isempty(report.orphaned_teams)
        @test isempty(report.orphaned_programs)
    end

    @testset "TPGMutationStrategy - mutate_program_action_rate" begin
        Random.seed!(111)
        tpg = TangledProgramGraph([1, 2, 3])
        p1 = TMAGE.add_program!(tpg, prog1_genome, action = 1)
        p2 = TMAGE.add_program!(tpg, prog2_genome, action = 2)
        team_target1 = TMAGE.add_team!(tpg, [p1.id])
        team_target2 = TMAGE.add_team!(tpg, [p2.id])
        set_root_team!(tpg, team_target2.id) # a root

        # Create the original team that will be copied and mutated
        # p1 points to team_target1. p2 is just a program in the team.
        original_team_action_map = Dict(p1.id => team_target1.id)
        original_team_progs = [p1.id, p2.id]
        original_team = TMAGE.add_team!(tpg, original_team_progs, original_team_action_map)
        original_team_id = original_team.id
        set_root_team!(tpg, original_team.id)

        original_tpg_teams_count = length(tpg.teams)
        original_tpg_program_count = length(tpg.programs)

        original_team_programs_set = Set([p.id for p in original_team.programs])
        original_team_action_map_copy = deepcopy(original_team.action_map)
        original_team_out_edges_copy = deepcopy(original_team.out_edges)

        # --- Evaluation before mutation ---
        input_data = [1, 1, 10]
        # Original Team behavior with input_data:
        # p1: 0.1 (action 1)
        # p2: 10. (action 2)
        # Winner is p2. p2 => action 2
        original_team_output_before_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_before_mutation == 2

        # 2. Configure Mutation
        config = TPGMutationConfig(
            0.0, # program_mutation_rate
            0.0, # add_program_to_team_rate
            0.0, # remove_program_from_team_rate
            0.0, # mutate_action_map_rate (FORCE MUTATE ACTION MAP)
            1.0  # mutate_program_action_rate
        )
        strategy = TPGMutationStrategy()

        # 3. Perform mutation
        TMAGE._mutate_single_offspring!(tpg, original_team_id, strategy, config, ma, ml, nc, si)
        # [ Info: Program P3: Action changed from 1 to 2.
        # [ Info: Removed program P1 from team T4.
        # [ Info: Program P4: Action changed from 2 to 3.
        # [ Info: Removed program P2 from team T4.

        # so p3 was made (copy of p1) in T4. p3 => action 2 (instead of 1)
        # so p4 was made (copy of p2) in T4. p4 => action 3 (instead of 2)

        # New state
        @test length(tpg.teams) == original_tpg_teams_count + 1 # One new team (the copy)
        @test length(tpg.programs) == original_tpg_program_count + 2 # 2 new, since p1 replaced by p3, p2 replaced by p4

        # Original team is unchanged
        retrieved_original_team = find_team_by_id(tpg, original_team_id)
        @test retrieved_original_team !== nothing
        @test Set([p.id for p in retrieved_original_team.programs]) == original_team_programs_set
        @test retrieved_original_team.action_map == original_team_action_map_copy
        @test retrieved_original_team.out_edges == original_team_out_edges_copy

        # Check the newly created (mutated) team
        new_team_id = setdiff(keys(tpg.teams), Set([original_team_id])) |> first
        new_team = find_team_by_id(tpg, new_team_id)
        @test new_team !== nothing
        @test Set([p.id for p in new_team.programs]) == Set((ProgramID(3), ProgramID(4)))

        # Identify the mutated action map entry:
        @test length(new_team.action_map) == 1
        # original action map was : P1 => T1 so now is P3 => T1
        @test new_team.action_map[ProgramID(3)] == TeamID(1) # new action map

        original_team_output_after_mutation, _ = evaluate(tpg, original_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test original_team_output_after_mutation == original_team_output_before_mutation # Original team output must be the same

        # p3 = 0.1, p4 = 10. P4 wins bid
        # p4 => action 3
        new_team_out, path = evaluate(tpg, new_team_id, input_data, si, ml, ma; cache_mode = PerInputCache)
        @test new_team_out == 3
        @test length(path) == 1

        # Test program in edges
        @test p1.in_edges == Set((TeamID(1), TeamID(3))) # new team does not has it
        @test p2.in_edges == Set((TeamID(2), TeamID(3))) # new team does not has it

        # Test program out edges
        p3 = find_program_by_id(tpg, ProgramID(3))
        p4 = find_program_by_id(tpg, ProgramID(4))
        @test p3.in_edges == Set([TeamID(4)])
        @test p4.in_edges == Set([TeamID(4)])
        @test p3.out_edges == Set((TeamID(1),))
        @test p4.out_edges |> isempty

        # New team out edge is empty
        @test new_team.in_edges |> isempty
        @test new_team.out_edges == Set([TeamID(1)])
        @test find_team_by_id(tpg, TeamID(1)).in_edges == Set((TeamID(4), TeamID(3)))

        # --- Integrity Check ---
        report = verify_tpg_integrity!(tpg; cleanup_orphans = false)
        @test report.is_consistent # Ensure no inconsistencies introduced
        @test isempty(report.orphaned_teams)
        @test isempty(report.orphaned_programs)
    end
end
