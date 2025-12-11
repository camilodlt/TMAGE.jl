using Random
using Statistics
using StatsBase
using Base.Threads

##############################################################
# EVOLUTIONARY ALGORITHM FOR TPGs                            #
##############################################################

function ea_train_tpg_mage(
        X::Any,
        num_initial_teams::Int,
        programs_per_initial_team::Int,
        generations::Int,
        num_offspring_per_gen::Int,
        tpg_mutation_config::TPGMutationConfig,
        ma::modelArchitecture,
        ml::MetaLibrary,
        nc::nodeConfig,
        si::SharedInput,
        initial_actions::Vector{A},
        fitness_calculator,
        fitness_type,
        metric_of_interest,
        endpoint_to_float,
        epoch_callbacks::UTCGP.Optional_FN,
        early_stop_callback::UTCGP.Optional_FN,
        warmup::Bool = true,
        cache::Union{Nothing, TPGEvaluationCache} = nothing,
        k::Int = 5 # how many to run on val ?
    ) where {A}

    if num_initial_teams <= 0
        error("`num_initial_teams` must be positive.")
    end
    if programs_per_initial_team <= 0
        error("`programs_per_initial_team` must be positive.")
    end
    if num_offspring_per_gen < 0
        error("`num_offspring_per_gen` cannot be negative.")
    end

    BatchSize = X.batch_size
    TrainSize = length(X)

    tpg = TangledProgramGraph(initial_actions)
    M_gen_loss_tracker = UTCGP.GenerationLossTracker()

    @info "Initializing TPG with $(num_initial_teams) root teams..."
    # 1. Initialization: Create initial root teams
    for _ in 1:num_initial_teams
        team_id = add_random_team_to_tpg!(tpg, ma, ml, nc, si; up_to_n_new_progs = programs_per_initial_team)
        set_root_team!(tpg, team_id)
    end
    @info "Initial TPG has $(length(tpg.teams)) teams and $(length(tpg.programs)) programs."

    for gen in 1:generations
        @info "--- Generation $(gen)/$(generations) ---"
        current_root_ids = collect(tpg.root_teams) # Get current elite root IDs

        new_offspring_root_ids = TeamID[]
        for _ in 1:num_offspring_per_gen
            parent_root_id = rand(tpg.root_teams) #rand(collect(keys(tpg.teams))) # Another option is to select one ROOT
            latest_team_id = _mutate_single_offspring!(tpg, parent_root_id, TPGMutationStrategy(), tpg_mutation_config, ma, ml, nc, si, cache)
            if haskey(tpg.teams, latest_team_id) && in(latest_team_id, tpg.root_teams) && !(latest_team_id in new_offspring_root_ids)
                push!(new_offspring_root_ids, latest_team_id)
            else
                error("New root is not in tpg or is not in roots : $latest_team_id")
            end
        end
        @info "Created $(length(new_offspring_root_ids)) new offspring root teams."

        # We now have way more root teams
        all_eval_root_ids = collect(tpg.root_teams)

        # 3. Evaluate all root nodes
        # sample_inputs_batch, _ = zip(X.xs, X.ys) #sample(X, TrainSize) # sample a mini batch
        sample_inputs_batch, _ = sample(X, TrainSize) # sample a mini batch
        @info "Evaluating $(length(all_eval_root_ids)) individuals on $(length(sample_inputs_batch)) samples..."
        fitness_matrix = Matrix{fitness_type}(undef, length(all_eval_root_ids), length(sample_inputs_batch))

        # which programs does not have cache ?
        ps_without_cache = programs_without_cache(cache, tpg) # prints how many programs don't have a cache
        if warmup && length(ps_without_cache) > nthreads()
            @assert cache.mode == LRUCacheMode "Only LRU is thread safe for warmup"
            programs = [find_program_by_id(tpg, pid) for pid in ps_without_cache]
            n_progs = length(programs)
            programs_per_thread = ceil(Int, n_progs / nthreads())
            tasks = []
            for partition in Iterators.partition(1:n_progs, programs_per_thread)
                indices_for_thread = collect(partition)
                t = Threads.@spawn begin
                    @info "BEGIN In thread $(threadid()) evaluating progs idx $partition"
                    for prog_idx in indices_for_thread
                        tpg_program = programs[prog_idx]
                        for (x, y) in sample_inputs_batch
                            evaluate(tpg_program, x, cache, si, ml, ma)
                        end
                    end
                    @info "DONE In thread $(threadid()) evaluating progs idx $partition"
                end
                push!(tasks, t)
            end
            @info "Fetching loading tasks"
            fetch.(tasks)
        end

        for (i, root_id) in enumerate(all_eval_root_ids)
            individual_fitnesses = fitness_type[]
            # individual_outs = Int[]
            for (sample_input, sample_gt) in sample_inputs_batch
                output, _ = evaluate(tpg, root_id, sample_input, si, ml, ma; cache = cache)
                f = fitness_calculator(output, sample_gt)
                push!(individual_fitnesses, f)
            end
            fitness_matrix[i, :] = individual_fitnesses
        end
        Metrics_per_individual = endpoint_to_float(fitness_matrix, fitness_calculator)
        ind_performances = map(x -> x[:loss], Metrics_per_individual)
        ind_baccs = map(x -> x[:bacc], Metrics_per_individual)

        # Calculate mean fitness for each individual across samples
        mean_fitnesses = vec(mean(ind_performances, dims = 2))

        # Find best individual in this generation
        best_fitness_gen, best_idx_gen = findmin(mean_fitnesses)
        best_bacc = ind_baccs[best_idx_gen]
        best_root_id_gen = all_eval_root_ids[best_idx_gen]
        UTCGP.affect_fitness_to_loss_tracker!(M_gen_loss_tracker, gen, best_fitness_gen)
        @warn "Generation $(gen): Best Fitness = $(best_fitness_gen) (Bacc:$best_bacc) (Root $(best_root_id_gen))"

        # 4. Selection: Remove a percentage of lowest-fitness individuals
        num_to_keep = num_initial_teams

        # Sort individuals by fitness (lower is better) and get the IDs to keep
        sorted_indices = sortperm(mean_fitnesses)
        elite_indices = sorted_indices[1:num_to_keep]
        elite_root_ids_next_gen = Set(all_eval_root_ids[elite_indices])
        non_elite_root_ids = setdiff(Set(all_eval_root_ids), elite_root_ids_next_gen)

        # 10 best roots
        ten_best_indices = sorted_indices[1:k]
        ten_elite_roots_ids = Set(all_eval_root_ids[ten_best_indices])

        # 5. Integrity Check & GC: Remove non-elite roots and clean the graph
        @info "Removing $(length(non_elite_root_ids)) non-elite roots..."
        for non_elite_id in non_elite_root_ids
            delete!(tpg.root_teams, non_elite_id) # Unset as root
        end

        @info "Running garbage collection..."
        initial_teams = length(tpg.teams)
        initial_programs = length(tpg.programs)
        report = verify_tpg_integrity!(tpg; cleanup_orphans = true, cache = cache)
        teams_removed_gc = initial_teams - length(tpg.teams)
        programs_removed_gc = initial_programs - length(tpg.programs)

        @info "GC Report: $(report.is_consistent ? "Consistent" : "Inconsistent") after cleanup."
        @info "Teams removed by GC: $(teams_removed_gc), Programs removed by GC: $(programs_removed_gc)."
        @info report

        if !report.is_consistent
            @warn "TPG is inconsistent after GC. This should not happen. Terminating EA."
            break
        end

        # 6. Callbacks: Optional epoch_callback
        if epoch_callbacks !== nothing
            @info "Calling epoch callback..."
            for epoch_fn in epoch_callbacks
                epoch_fn(
                    (
                        iteration = gen, tpg = tpg, best_root = best_root_id_gen, best_fitness = best_fitness_gen,
                        best_roots = ten_elite_roots_ids, best_fitnesses = mean_fitnesses[ten_best_indices],
                        cache = cache,
                        si = si, ma = ma, ml = ml, fitness_type = fitness_type,
                    )
                )
            end
        end

        if !isnothing(early_stop_callback)
            to_break = early_stop_callback()
            if to_break
                break
            end
        end
    end

    # After generations, ensure one final GC run
    @info "Final garbage collection after EA loop."
    final_report = verify_tpg_integrity!(tpg; cleanup_orphans = true, cache = cache)
    if !final_report.is_consistent
        @warn "TPG is inconsistent after final GC."
    end

    return tpg, M_gen_loss_tracker
end
